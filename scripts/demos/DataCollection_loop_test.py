"""
Collect G1 locomotion data toward waypoint with shortest yaw rotation.
- Robot can move in full 2D (vx, vy)
- Adjusts yaw_rate smoothly using shortest-angle correction
- Keeps all existing camera / data / marker logic
"""

import os
import csv
import sys
import torch
import numpy as np
import argparse
from datetime import datetime

# ---------------------------------------------------------------------
# Isaac Lab launcher
# ---------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect G1 locomotion data with waypoint and yaw fix.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--no_collect", action="store_true", help="Disable data collection (only visualize robot).")
parser.add_argument("--waypoint_x", type=float, default=2.0, help="Waypoint X position in world coordinates.")
parser.add_argument("--waypoint_y", type=float, default=1.0, help="Waypoint Y position in world coordinates.")
parser.add_argument("--vx", type=float, default=0.5, help="Initial vx command (m/s).")
parser.add_argument("--vy", type=float, default=0.0, help="Initial vy command (m/s).")
parser.add_argument("--yaw_rate", type=float, default=0.0, help="Initial yaw rate (rad/s).")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------
# Imports after launching app
# ---------------------------------------------------------------------
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.sensors import ContactSensorCfg

import isaaclab.sim as sim_utils
from pxr import UsdGeom, Gf, Sdf
import omni.usd

ISAACLAB_LEG_IDXS = torch.tensor([
    0, 3, 7, 11, 15, 19,
    1, 4, 8, 12, 16, 20
])

class G1TurningCollector:
    """Collect locomotion data with correct yaw-angle normalization."""

    @staticmethod
    def _waypoints_to_list(waypoint) -> list:
        """Single point (2,) -> [pt]; multiple (N,2) -> list of N points.

        Do not use ``isinstance(wp[0], float)``: ``np.float32`` is not a Python float,
        so a length-2 array would be mistaken for two separate scalars and break indexing.
        """
        w = np.asarray(waypoint, dtype=np.float64)
        if w.ndim == 1 and w.size == 2:
            return [w]
        if w.ndim == 2 and w.shape[1] == 2:
            return [w[i] for i in range(w.shape[0])]
        raise ValueError(f"waypoint must be shape (2,) or (N, 2), got {w.shape}")

    def __init__(self, vx=0.5, vy=0.0, yaw_rate=0.0,
                 waypoint=(2.0, 1.0), img_res=(640, 480),
                 save_every=10, collect_data=True):
        TASK = "Isaac-Velocity-Flat-G1-v0"
        RL_LIBRARY = "rsl_rl"
        self.collect_data = collect_data
        self.waypoint = np.array(waypoint)
        # Obstacle center on XY plane (used for region definition).
        self.obstacle_xy = np.array([2.0, 0.0], dtype=np.float64)
        # Four circular regions: front / back / left / right.
        # You only need to tune center and radius values below.
        self.trajectory_regions = {
            "front": {"center": np.array([0.0, 0], dtype=np.float64), "r": 0.5},
            "back": {"center": np.array([3, 0], dtype=np.float64), "r": 0.25},
            "left": {"center": np.array([2.0, 1.2], dtype=np.float64), "r": 0.35},
            "right": {"center": np.array([2.0, -1.2], dtype=np.float64), "r": 0.35},
        }

        # --- RL config & checkpoint ---
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)

        # --- Environment ---
        env_cfg = G1FlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 100000
        env_cfg.curriculum = None
        env_cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0) 
        # End-to-end 200 Hz control/data loop: dt=0.005, decimation=1.
        env_cfg.decimation = 1
        env_cfg.sim.render_interval = 1

        # disable torso-contact termination so collisions with objects don't reset the env
        env_cfg.terminations.base_contact = None ###WE ADDED THIS HERE TO FIX ENV RESET WHEN WE HIT TORSO OF HUMANOID AND OBSTALE

        # --- Add Obstacle below ---
        # self._add_obstacle_cube(env_cfg, pos=(2, 0.0, 5.25), size=(0.5, 1.0, 0.5),index=0)
        #self._add_blue_bin(env_cfg, pos=(2, 0, 0.25),index=0)
        self._add_table(env_cfg, pos=(2, 0, 0.25),index=0)


        # --- Add Obstacle above ---
        
        env_cfg.scene.robot_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*link.*", 
            update_period=0.0,
            debug_vis=True,
            filter_prim_paths_expr=[],
        )
        
        
        # --- Sensor target rates (aligned to real robot) ---
        self.camera_fps = 15.0
        self.lidar_fps = 7.0
        self.camera_period_s = 1.0 / self.camera_fps
        self.lidar_period_s = 1.0 / self.lidar_fps

        # --- Add camera ---
        env_cfg.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_link/front_camera",
            update_period=self.camera_period_s,
            height=img_res[0],
            width=img_res[1],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),

            offset=CameraCfg.OffsetCfg(
                pos=(0.3, 0.0, 0.5),
                rot=(0.0, 0.924, 0.0, 0.383),   #  forward + 42° downward
                convention="ros",
            ),
        )

        # --- Add lidar ---
        env_cfg.scene.lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_link",   
            update_period=self.lidar_period_s,

            offset=RayCasterCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
            ),

            mesh_prim_paths=["/World"],   

            ray_alignment="yaw",

            pattern_cfg=patterns.LidarPatternCfg(
                channels=32,                      # 垂直线数（先别太大）
                vertical_fov_range=(-20, 20),
                horizontal_fov_range=(-180, 180),
                horizontal_res=2.0,              # 分辨率（deg）
            ),

            debug_vis=False,
        )

        # --- Create environment ---
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        self.sim_dt = float(self.env.unwrapped.cfg.sim.dt)
        self.sim_hz = 1.0 / self.sim_dt
        print(f"[INFO] Control/data loop set to {self.sim_hz:.1f} Hz (dt={self.sim_dt:.4f}s)")
        print(f"[INFO] Camera/LiDAR target rates: {self.camera_fps:.1f} Hz / {self.lidar_fps:.1f} Hz")
        self.next_camera_time_s = 0.0
        self.next_lidar_time_s = 0.0
        self.camera_frame_idx = 0
        self.lidar_frame_idx = 0
        # load custom scene
        self._load_scene_usd()

        # --- Load pretrained policy ---
        runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(checkpoint)
        self.policy = runner.get_inference_policy(device=self.device)

        # --- Velocity command ---
        self.commands = torch.zeros(1, 3, device=self.device)
        self.commands[:, 0] = vx
        self.commands[:, 1] = vy
        self.commands[:, 2] = yaw_rate

        self.max_speed = float(np.linalg.norm([vx, vy]))
        self.save_every = save_every

        # --- Output dirs: data/<timestamp>/ = one trajectory run ---
        self.data_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
        self.base_dir = ""
        self.image_dir = ""
        self.lidar_dir = ""
        self.save_path = ""
        self._create_new_trajectory_output()
        self.dataset_file = None
        self.contact_file = None
        self.contact_writer = None

        # --- Robot info ---
        robot = self.env.unwrapped.scene["robot"]
        self.num_joints = robot.data.joint_pos.shape[1]
        print(f"[INFO] Detected {self.num_joints} actuated joints. Waypoint = {self.waypoint}")
        #print(f"[INFO] Detected {self.num_joints} actuated joints. Waypoint = {self.waypoint}")

        # --- Camera handle ---
        self.camera = self.env.unwrapped.scene["camera"]
        print(f"[INFO] Camera initialized. Data collection = {self.collect_data}")

        # --- lidar handle (directory already created above) ---

        # --- Add waypoint marker (green sphere) ---
        self._add_waypoint_marker()

        # --- Add obstacle cube ---
        #self._add_obstacle_cube(pos=(2.0, 0.0, 0.5), size=1.0)

    def _load_scene_usd(self):
        stage = omni.usd.get_context().get_stage()

        scene_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scene_new/lab.usda")
        )

        prim_path = "/World/ExternalScene"

        if stage.GetPrimAtPath(prim_path):
            print("[INFO] Scene already exists")
            return

        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(scene_path)
    

        # ---------- control scene transform ----------
        xform = UsdGeom.Xformable(prim)

        xform.AddTranslateOp().Set(Gf.Vec3f(2, -1, 1.85)) # 2 , -1
        xform.AddRotateZOp().Set(50)     
        xform.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
        #print(xform.GetLocalTransformation())
        # -----------------------------------------

        #print("[INFO] Scene loaded")

        # remove default ground
        ground_path = "/World/ground"
        if stage.GetPrimAtPath(ground_path):
            stage.RemovePrim(ground_path)



    def _add_obstacle_cube(self, env_cfg, pos, size, index):
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg

        name = f"obstacle_cube_{index}"

        setattr(
            env_cfg.scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.CuboidCfg(
                    size=size,

                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    ),

                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)
                    ),
                ),

                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
        )

        #print(f"[INFO] Added {name} at {pos}")

    def _add_blue_bin(self, env_cfg, pos, index):

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.converters import MeshConverterCfg, MeshConverter
        from isaaclab.sim.schemas import schemas_cfg

        name = f"blue_bin_{index}"

        glb_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scene_new/blue_bin.glb")
        )

        usd_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scene_new/_converted_blue_bin")
        )

        converter = MeshConverter(MeshConverterCfg(
            asset_path=glb_path,
            usd_dir=usd_dir,
            make_instanceable=True,
            force_usd_conversion=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            ),
        ))

        #print(f"[INFO] Converted blue_bin.glb → {converter.usd_path}")

        setattr(
            env_cfg.scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=converter.usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    scale=(0.65, 0.65, 0.65),
                ),
                # Quaternion is (w, x, y, z):
                # keep upright (original qx90) and rotate heading by 90deg.
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(0.5, 0.5, 0.5, 0.5)),
            )
        )
        #print(f"[INFO] Added blue bin at {pos}")


    def _add_table(self, env_cfg, pos, index):

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.converters import MeshConverterCfg, MeshConverter
        from isaaclab.sim.schemas import schemas_cfg

        name = f"table_{index}"

        glb_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scene_new/table.glb")
        )

        usd_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scene_new/_converted_table1")
        )

        converter = MeshConverter(MeshConverterCfg(
            asset_path=glb_path,
            usd_dir=usd_dir,
            make_instanceable=True,
            force_usd_conversion=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
            mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            ),
        ))

        #print(f"[INFO] Converted blue_bin.glb → {converter.usd_path}")

        setattr(
            env_cfg.scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=converter.usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
                    scale=(1, 1, 1),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos,rot=(0.5, 0.5, 0.5, 0.5)),
            )
        )
        #print(f"[INFO] Added table at {pos}")



    def _add_waypoint_marker(self):
        """Add green sphere markers for all waypoints."""
        stage = self.env.unwrapped.scene.stage

        waypoints = self._waypoints_to_list(self.waypoint)

        for i, wp in enumerate(waypoints):
            sphere_path = Sdf.Path(f"/World/WaypointMarker_{i}")
            if stage.GetPrimAtPath(sphere_path):
                continue 
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.GetRadiusAttr().Set(0.1)
            sphere.AddTranslateOp().Set(Gf.Vec3f(wp[0], wp[1], 0.0))
            sphere.CreateVisibilityAttr().Set("invisible")

            #color_attr = sphere.CreateDisplayColorAttr()
            #color_attr.Set([(0.0, 1.0, 0.0)])  
        print(f"[INFO] Added {len(waypoints)} waypoint markers.")

    def _sample_point_in_region(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Uniformly sample a point inside a 2D circle."""
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        rr = radius * np.sqrt(np.random.uniform(0.0, 1.0))
        return center + rr * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)

    def _generate_random_waypoint_sequence(self) -> np.ndarray:
        """front -> (left/right) -> back; each point sampled in circular region."""
        front_cfg = self.trajectory_regions["front"]
        back_cfg = self.trajectory_regions["back"]
        side_name = np.random.choice(["left", "right"])
        side_cfg = self.trajectory_regions[side_name]

        front_pt = self._sample_point_in_region(front_cfg["center"], float(front_cfg["r"]))
        side_pt = self._sample_point_in_region(side_cfg["center"], float(side_cfg["r"]))
        back_pt = self._sample_point_in_region(back_cfg["center"], float(back_cfg["r"]))

        waypoints = np.stack([front_pt, side_pt, back_pt], axis=0)
        print(f"[INFO] New trajectory waypoints (side={side_name}): {waypoints}")
        return waypoints

    def _create_new_trajectory_output(self):
        """Create a fresh timestamped folder for one trajectory."""
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms resolution
        self.base_dir = os.path.join(self.data_parent, run_stamp)
        os.makedirs(self.base_dir, exist_ok=True)
        self.image_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.lidar_dir = os.path.join(self.base_dir, "lidar")
        os.makedirs(self.lidar_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "locomotion_dataset.csv")
        print(f"[INFO] Trajectory folder: {self.base_dir}")

    def _open_data_writers(self, header):
        """Open csv writers for the current trajectory folder."""
        self.dataset_file = open(self.save_path, mode="w", newline="")
        writer = csv.writer(self.dataset_file)
        writer.writerow(header)

        sensor = self.env.unwrapped.scene["robot_contact"]
        link_names = sensor.body_names
        self.contact_file = open(
            os.path.join(self.base_dir, "contact_force.csv"),
            "w", newline=""
        )
        self.contact_writer = csv.writer(self.contact_file)
        contact_header = ["step"]
        for name in link_names:
            contact_header += [f"{name}_x", f"{name}_y", f"{name}_z"]
        self.contact_writer.writerow(contact_header)
        return writer

    def _close_data_writers(self):
        """Close all active trajectory files safely."""
        if self.dataset_file is not None:
            self.dataset_file.close()
            self.dataset_file = None
        if self.contact_file is not None:
            self.contact_file.close()
            self.contact_file = None
        self.contact_writer = None


    def quat_to_yaw(self, quat):
        w, x, y, z = quat  
        ##print(f"x:{x},y:{y},z:{z},w:{w}")
        yaw = np.arctan2(2.0 * (w * z + x * y),
                        1.0 - 2.0 * (y * y + z * z))
        return yaw
    


    def normalize_angle(self, angle):
        """Wrap angle into [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    
    # -----------------------------------------------------------------
    #  Main loop (dynamic alignment world→local, auto-reset until stop)
    # -----------------------------------------------------------------
    
    def run(self, num_steps=3000):
        obs, _ = self.env.reset()

        # ---- initial facing toward +X ----
        scene = self.env.unwrapped.scene
        robot = scene["robot"]

        # Generate waypoints per trajectory (dynamic each reset).
        self.waypoint = self._generate_random_waypoint_sequence()
        waypoints = self._waypoints_to_list(self.waypoint)
        start_xy = np.array(waypoints[0], dtype=np.float64)
        root_pose = torch.tensor([[float(start_xy[0]), float(start_xy[1]), 0.8, 1.0, 0.0, 0.0, 0.0]], device=self.device)
        robot.write_root_pose_to_sim(root_pose)
        self._add_waypoint_marker()

        print(f"[INFO] Running {num_steps} steps through {len(waypoints)} waypoints: {waypoints}")

        stop_thresh = 0.1
        k_yaw = 1.0
        max_yaw_rate = 1.0
        yaw_smooth = 0.1
        prev_yaw_rate = 0.0
        current_target_idx = 0
        threshold_deg = 55
        target = np.array(waypoints[current_target_idx])
        trajectory_count = 1
        trajectory_step = 0

        prev_yaw = 0.0
        prev_theta_v = 0.0

        # --- Data collection ---
        if self.collect_data:
            N = self.num_joints
            header = (
                ["sim_step", "sim_time_s"] +
                ["px", "py", "pz"] +
                [f"base_quat_{i}" for i in range(4)] +
                ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"] +
                ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"] +
                [f"joint_pos_{i}" for i in range(N)] +
                [f"joint_vel_{i}" for i in range(N)] +
                [f"torque_{i}" for i in range(N)] +
                [f"action_{i}" for i in range(N)] +
                ["vx_cmd", "vy_cmd", "yaw_rate_cmd"] +
                ["target_x", "target_y"]
            )
            writer = self._open_data_writers(header)

        else:
            writer = None

        # ======================================================
        # Main loop
        # ======================================================
        for step in range(num_steps):
            robot = self.env.unwrapped.scene["robot"]
            data = robot.data
            
            
            base_pos = data.root_pos_w[0].cpu().numpy()
            base_quat = data.root_quat_w[0].cpu().numpy()

            # unwrap yaw
            yaw = self.quat_to_yaw(base_quat)
            yaw = np.unwrap([prev_yaw, yaw])[1]
            prev_yaw = yaw

            # waypoint vector
            dx = target[0] - base_pos[0]
            dy = target[1] - base_pos[1]
            dist = np.sqrt(dx**2 + dy**2)

            # waypoint reached?
            if dist < stop_thresh:
                #print(f"[INFO] Reached waypoint {current_target_idx+1}/{len(waypoints)} at step {step}, dist={dist:.3f}")
                current_target_idx += 1
                if current_target_idx >= len(waypoints):
                    print(f"[INFO] Trajectory #{trajectory_count} completed at global step {step}. Reset to start.")
                    trajectory_count += 1
                    current_target_idx = 0

                    # reset robot/environment state to start the next trajectory
                    obs, _ = self.env.reset()
                    self.waypoint = self._generate_random_waypoint_sequence()
                    waypoints = self._waypoints_to_list(self.waypoint)
                    target = np.array(waypoints[current_target_idx])
                    start_xy = np.array(waypoints[0], dtype=np.float64)
                    root_pose = torch.tensor(
                        [[float(start_xy[0]), float(start_xy[1]), 0.8, 1.0, 0.0, 0.0, 0.0]],
                        device=self.device
                    )
                    robot.write_root_pose_to_sim(root_pose)
                    self.commands[:] = 0.0
                    prev_yaw_rate = 0.0
                    prev_yaw = 0.0
                    prev_theta_v = 0.0
                    trajectory_step = 0
                    self._add_waypoint_marker()

                    if self.collect_data:
                        assert writer is not None
                        self._close_data_writers()
                        self._create_new_trajectory_output()
                        writer = self._open_data_writers(header)
                        self.camera_frame_idx = 0
                        self.lidar_frame_idx = 0
                        self.next_camera_time_s = 0.0
                        self.next_lidar_time_s = 0.0
                    continue
                else:
                    target = np.array(waypoints[current_target_idx])
                    #print(f"[INFO] Switching to next waypoint: {target}")
                    self.waypoint = target
                    self._add_waypoint_marker()
                    continue

            # world → local transform
            local_dx =  np.cos(yaw)*dx + np.sin(yaw)*dy
            local_dy = -np.sin(yaw)*dx + np.cos(yaw)*dy

            direction_local = np.array([local_dx, local_dy])
            direction_local /= np.linalg.norm(direction_local)
            

            # ideal vx, vy
            vx_local = self.max_speed * direction_local[0]
            vy_local = self.max_speed * direction_local[1]

            # ----------------------------
            # compute θ_v from velocity
            # ----------------------------
            theta_v = np.arctan2(vy_local, vx_local)

            # wrap [-pi, pi]
            theta_v = (theta_v + np.pi) % (2*np.pi) - np.pi

            #print(f"theta_v: {np.degrees(theta_v)}")

            # -----------------------------------------------
            # 
            # -----------------------------------------------
            theta_deg = np.degrees(theta_v)

            dead_zone_deg = 30       
            dead_zone_start = 180 - dead_zone_deg

            if -threshold_deg <= theta_deg <= threshold_deg:
                # move only
                yaw_rate_to_use =  k_yaw * theta_v
                vx_cmd = vx_local
                vy_cmd = vy_local

            else:
                # 
                vx_cmd = 0.1
                vy_cmd = 0.0     
                yaw_smooth = 1.0

                # ==========================================
                # back dead-zone：(yaw_rate）
                # ==========================================
                if abs(theta_deg) >= dead_zone_start:
                    # ±dead_zone
                    yaw_rate_to_use = +max_yaw_rate     # always turn left
                    # always turn right yaw_rate_to_use = -max_yaw_rate

                else:
                    # regular misaligned but not in dead-zone
                    yaw_rate_to_use = k_yaw * theta_v
                    yaw_rate_to_use = np.clip(yaw_rate_to_use, -max_yaw_rate, max_yaw_rate)


            # smooth yaw
            yaw_rate = (1 - yaw_smooth) * prev_yaw_rate + yaw_smooth * yaw_rate_to_use
            prev_yaw_rate = yaw_rate

            # ===== Smooth linear velocity commands (VERY IMPORTANT) =====
            alpha = 1   # smoothing factor 0.1~0.3
            prev_vx = self.commands[0,0].item()
            prev_vy = self.commands[0,1].item()

            vx_cmd = (1 - alpha) * prev_vx + alpha * vx_cmd
            vy_cmd = (1 - alpha) * prev_vy + alpha * vy_cmd
            # ============================================================


            # update commands
            target_cmd = torch.tensor([[vx_cmd, vy_cmd, yaw_rate]], device=self.device)
            if step == 0:
                self.commands = target_cmd.clone()

            self.commands = target_cmd.clone()

            self.env.unwrapped.command_manager._terms["base_velocity"].command[:] = self.commands.clone()
            obs["policy"][0, 11] = self.commands[0, 2]


            print(f"obs: {obs}")
            vec = obs["policy"][0]

            print("base_lin_vel:", vec[0:3])
            print("base_ang_vel:", vec[3:6])
            print("proj_gravity:", vec[6:9])
            print("commands:", vec[9:12])
            print("joint_pos:", vec[12:49])
            print("joint_vel:", vec[49:86])
            print("actions:", vec[86:123])
            # RL policy
            with torch.inference_mode():
                # write to env

                actions = self.policy(obs)
                print(f"actions: {actions}")

            #idx12 = [0,1,3,4,7,8,11,12,15,16,19,20]
            #actions_new = torch.zeros_like(actions)
            #for i,new_i in enumerate(idx12):
                #actions_new[:, new_i] = actions[:, new_i]   # 保留这12维
            #print(f"action_new: {actions_new}")
            
            obs, _, _, _ = self.env.step(actions)


            # ===== CONTACT =====
            sensor = self.env.unwrapped.scene["robot_contact"]

            print("Contact Force:")
            print(sensor.body_names)
            print(sensor.data.net_forces_w)

            # =========================
            # LIDAR COLLECTION 
            # =========================
            lidar = self.env.unwrapped.scene["lidar"]

            lidar_points = lidar.data.ray_hits_w[0]   # (N_rays, 3)
            lidar_np = lidar_points.detach().cpu().numpy()
            lidar_np = np.nan_to_num(lidar_np, nan=0.0, posinf=0.0, neginf=0.0)


            print(f"lidar_np: {lidar_np}")
            print(f"base pose: {base_pos}")

            # 转 range
            #origin = base_pos
            #ranges = np.linalg.norm(lidar_np - origin, axis=1)

            ranges = np.linalg.norm(lidar_np, axis=1)
            print(f"ranges: {ranges}")

            # debug 
            print("lidar shape:", lidar_np.shape)
            print("lidar min/max:", ranges.min(), ranges.max())
                

            # Locomotion printing
            print(f"[STEP {step}] Target={target}, dist={dist:.2f}, yaw={np.degrees(yaw):.1f}°, "
                f"vx={vx_cmd:.2f}, vy={vy_cmd:.2f}, yaw_rate={np.degrees(yaw_rate):.1f}°/s")

            # save data
            if self.collect_data:
                assert writer is not None
                base_lin_vel = data.root_lin_vel_w[0].cpu().numpy()
                base_ang_vel = data.root_ang_vel_w[0].cpu().numpy()
                joint_pos = data.joint_pos[0, :self.num_joints].cpu().numpy()
                joint_vel = data.joint_vel[0, :self.num_joints].cpu().numpy()
                torques = data.applied_torque[0, :self.num_joints].cpu().numpy()
                actions_np = actions[0,:self.num_joints].detach().cpu().numpy()
                commands_np = self.commands[0].detach().cpu().numpy()
                sim_step = float(trajectory_step)
                sim_time_s = trajectory_step * self.sim_dt

                row = np.concatenate([
                    np.array([sim_step, sim_time_s], dtype=np.float64),
                    base_pos, base_quat, base_lin_vel, base_ang_vel,
                    joint_pos, joint_vel, torques, actions_np, commands_np, target
                ])
                writer.writerow(row.tolist())


                # ===== CONTACT SAVE =====
                contact = sensor.data.net_forces_w[0].detach().cpu().numpy()

                contact_row = [step]
                for i in range(contact.shape[0]):
                    contact_row += contact[i].tolist()

                assert self.contact_writer is not None
                self.contact_writer.writerow(contact_row)


                # -------- Camera SAVE (15 FPS) --------
                if sim_time_s + 1e-12 >= self.next_camera_time_s:
                    rgb_tensor = self.camera.data.output["rgb"][0]
                    rgb_np = rgb_tensor[..., :3].cpu().numpy()

                    if rgb_np.dtype != np.uint8:
                        rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)

                    rgb_np = np.rot90(rgb_np, k=1)

                    import imageio
                    imageio.imwrite(
                        os.path.join(self.image_dir, f"rgb_{self.camera_frame_idx:06d}.png"),
                        rgb_np,
                    )
                    self.camera_frame_idx += 1
                    self.next_camera_time_s += self.camera_period_s

                # -------- LIDAR SAVE (7 FPS) --------
                if sim_time_s + 1e-12 >= self.next_lidar_time_s:
                    np.save(
                        os.path.join(self.lidar_dir, f"lidar_{self.lidar_frame_idx:06d}.npy"),
                        ranges,
                    )
                    self.lidar_frame_idx += 1
                    self.next_lidar_time_s += self.lidar_period_s
            trajectory_step += 1
        
        if self.collect_data and writer is not None:
            self._close_data_writers()
            print(f"[INFO] Trajectory folder: {os.path.abspath(self.base_dir)}")
            print(f"[INFO] Dataset saved to: {os.path.abspath(self.save_path)}")
            print(f"[INFO] Images written to: {os.path.abspath(self.image_dir)}")
            print(f"[INFO] LiDAR written to: {os.path.abspath(self.lidar_dir)}")


def main():
    collect_flag = not args_cli.no_collect

    collector = G1TurningCollector(
    vx=args_cli.vx,
    vy=args_cli.vy,
    yaw_rate=args_cli.yaw_rate,
    waypoint=[(0,0),(1, 0), (2, 1), (3,0)], 
    img_res=(640, 480),
    save_every=1,
    collect_data=collect_flag,
    )
    
       
    collector.run(num_steps=60000)
    simulation_app.close()


if __name__ == "__main__":
    main()
