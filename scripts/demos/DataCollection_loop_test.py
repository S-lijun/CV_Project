
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
parser.add_argument(
    "--lidar_mesh_0",
    type=str,
    default="/World/envs/env_0/chair_0/geometry/mesh",
    help="First mesh prim path for RayCaster lidar_0 (one mesh per Isaac RayCaster).",
)
parser.add_argument(
    "--lidar_mesh_1",
    type=str,
    default="/World/envs/env_0/chair_0/geometry/mesh",
    help="Second mesh prim path for RayCaster lidar_1.",
)
parser.add_argument(
    "--lidar_mesh_paths",
    type=str,
    nargs="+",
    default=None,
    help="Optional list of mesh prim paths. If set, overrides --lidar_mesh_0/1.",
)
parser.add_argument(
    "--lidar_objects",
    type=str,
    nargs="+",
    default=None,
    help="Object names like chair_0 table_0; auto-converted to <env_prim_root>/<name>/geometry/mesh.",
)
parser.add_argument(
    "--env_prim_root",
    type=str,
    default="/World/envs/env_0",
    help="Root prim path used when --lidar_objects are provided.",
)
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

from data_collection_obstacles import add_chair, add_blue_bin, add_table

ISAACLAB_LEG_IDXS = torch.tensor([
    0, 3, 7, 11, 15, 19,
    1, 4, 8, 12, 16, 20
])


def _object_to_mesh_path(object_name: str, env_prim_root: str) -> str:
    """Convert obstacle object name to mesh prim path."""
    obj_name = object_name.strip()
    if not obj_name:
        raise ValueError("Empty object name in --lidar_objects is not allowed.")
    if obj_name.startswith("/"):
        return obj_name
    return f"{env_prim_root.rstrip('/')}/{obj_name}/geometry/mesh"


def _merge_ray_hits_multi(
    origin_np: np.ndarray,
    hits_list: list[np.ndarray],
    max_d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge k (N,3) world hit arrays from sensors with same origin."""
    if len(hits_list) == 0:
        raise ValueError("hits_list must contain at least one lidar hit array.")

    diffs = [hits - origin_np for hits in hits_list]
    ranges_all = [np.linalg.norm(d, axis=1) for d in diffs]
    finite_all = [
        np.isfinite(hits).all(axis=1) & (r > 1e-4) & (r < max_d * 0.999)
        for hits, r in zip(hits_list, ranges_all)
    ]

    n = hits_list[0].shape[0]
    merged = np.full_like(diffs[0], np.inf)
    best_r = np.full(n, np.inf)
    for diff, r, valid in zip(diffs, ranges_all, finite_all):
        take = valid & (r < best_r)
        merged[take] = diff[take]
        best_r[take] = r[take]

    diff_w = merged
    ranges = np.linalg.norm(diff_w, axis=1)
    ranges_xy = np.linalg.norm(diff_w[:, :2], axis=1)
    return diff_w, ranges, ranges_xy


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
        # Circular regions: each value is {"center": (2,) xy, "r": radius}.
        # Only keys listed in ``trajectory_region_sequence`` are sampled each episode.
        self.trajectory_regions = {
            "front": {"center": np.array([0.0, 0], dtype=np.float64), "r": 0.5},
            "back": {"center": np.array([3, 0], dtype=np.float64), "r": 0.3},
            "left": {"center": np.array([2, 0.5], dtype=np.float64), "r": 0.2},
            "right": {"center": np.array([2, -0.5], dtype=np.float64), "r": 0.2},
            # Example extra region (uncomment and add to trajectory_region_sequence to use):
            # "front2": {"center": np.array([1.0, 0.0], dtype=np.float64), "r": 0.4},
        }
        # Episode waypoint order: str = fixed region name; tuple[str, ...] = pick one name at random.
        # Default matches old behavior: front -> (left or right) -> back.
        self.trajectory_region_sequence = [
            "front",
            ("left", "right"),
            "back",
        ]
        if args_cli.lidar_objects is not None and len(args_cli.lidar_objects) > 0:
            mesh_paths = tuple(
                _object_to_mesh_path(obj_name, args_cli.env_prim_root)
                for obj_name in args_cli.lidar_objects
            )
        elif args_cli.lidar_mesh_paths is not None and len(args_cli.lidar_mesh_paths) > 0:
            mesh_paths = tuple(args_cli.lidar_mesh_paths)
        else:
            mesh_paths = (args_cli.lidar_mesh_0, args_cli.lidar_mesh_1)
        self._lidar_mesh_paths = mesh_paths

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

        # --- Add obstacles (see data_collection_obstacles.py) ---
        # add_obstacle_cube(env_cfg, pos=(2, 0.0, 5.25), size=(0.5, 1.0, 0.5), index=0)
        add_blue_bin(env_cfg, pos=(2, 0, 0.5), index=0)
        #add_table(env_cfg, pos=(2, 0, 0.5), index=0)
        #add_chair(env_cfg, pos=(2, 0, 0.5), index=0)
        
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

        _lidar_common = dict(
            prim_path="{ENV_REGEX_NS}/Robot/head_link",
            update_period=self.lidar_period_s,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=45,
                vertical_fov_range=(-90, 90),
                horizontal_fov_range=(-180, 180),
                horizontal_res=2.0,
            ),
            debug_vis=False,
        )
        self._lidar_sensor_names = []
        for i, mesh_path in enumerate(self._lidar_mesh_paths):
            sensor_name = f"lidar_{i}"
            setattr(env_cfg.scene, sensor_name, RayCasterCfg(mesh_prim_paths=[mesh_path], **_lidar_common))
            self._lidar_sensor_names.append(sensor_name)
            print(f"[INFO] LiDAR mesh path: {sensor_name} -> {mesh_path}")

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
        self._ignored_collision_links = {"left_ankle_roll_link", "right_ankle_roll_link"}
        self._collision_force_threshold = 0.1

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

        # --- Obstacles: register before env in __init__ via data_collection_obstacles.add_* ---
        self._obstacle_names = ("blue_bin_0", "table_0", "chair_0", "obstacle_cube_0")
        self._obstacle_default_states = {}
        self._cache_obstacle_default_states()

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

    def _cache_obstacle_default_states(self):
        """Cache default root states for known obstacles present in the scene."""
        scene = self.env.unwrapped.scene
        for name in self._obstacle_names:
            try:
                obstacle = scene[name]
            except KeyError:
                continue
            if not hasattr(obstacle, "data") or not hasattr(obstacle.data, "default_root_state"):
                continue
            self._obstacle_default_states[name] = obstacle.data.default_root_state.clone()
            print(f"[INFO] Cached default state for obstacle: {name}")

    def _reset_obstacles_to_default(self):
        """Restore obstacle pose/velocity to their default root states."""
        if not self._obstacle_default_states:
            return
        scene = self.env.unwrapped.scene
        for name, default_state in self._obstacle_default_states.items():
            try:
                obstacle = scene[name]
            except KeyError:
                continue
            obstacle.write_root_pose_to_sim(default_state[:, :7])
            if default_state.shape[1] >= 13 and hasattr(obstacle, "write_root_velocity_to_sim"):
                obstacle.write_root_velocity_to_sim(default_state[:, 7:13])

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
        """Sample one waypoint per entry in ``trajectory_region_sequence`` (see __init__)."""
        points: list[np.ndarray] = []
        log_names: list[str] = []

        for entry in self.trajectory_region_sequence:
            if isinstance(entry, str):
                region_name = entry
            elif isinstance(entry, tuple) and len(entry) > 0 and all(isinstance(x, str) for x in entry):
                region_name = str(np.random.choice(entry))
            else:
                raise TypeError(
                    "trajectory_region_sequence entries must be str or tuple[str, ...], "
                    f"got {type(entry).__name__}: {entry!r}"
                )

            if region_name not in self.trajectory_regions:
                raise KeyError(
                    f"Region {region_name!r} not in trajectory_regions. "
                    f"Keys: {list(self.trajectory_regions.keys())}"
                )
            cfg = self.trajectory_regions[region_name]
            pt = self._sample_point_in_region(cfg["center"], float(cfg["r"]))
            points.append(pt)
            log_names.append(region_name)

        waypoints = np.stack(points, axis=0)
        print(f"[INFO] New trajectory waypoints (regions={log_names}): {waypoints}")
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
        self._collision_link_indices = [
            i for i, name in enumerate(link_names)
            if name not in self._ignored_collision_links
        ]
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
        self._reset_obstacles_to_default()
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
        lidars = [self.env.unwrapped.scene[name] for name in self._lidar_sensor_names]

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
                ["target_x", "target_y"] +
                ["lidar_min_distance"] +
                ["lidar_min_distance_xy"] +
                ["is_collision"]
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
                    self._reset_obstacles_to_default()
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
            lidar_points_list = [lidar.data.ray_hits_w[0].detach().cpu().numpy() for lidar in lidars]
            origin_np = lidars[0].data.pos_w[0].detach().cpu().numpy()
            max_d = float(getattr(lidars[0].cfg, "max_distance", 1e6))

            diff_w, ranges, ranges_xy = _merge_ray_hits_multi(
                origin_np, lidar_points_list, max_d
            )
            finite_hit = np.isfinite(diff_w).all(axis=1)
            valid_ray = finite_hit & (ranges > 1e-4) & (ranges < max_d * 0.999)

            print(f"lidar merged (first 3 diff): {diff_w[:3]}")
            print(f"lidar origin_w: {origin_np}, base_pos: {base_pos}")
            print(f"ranges (first 3): {ranges[:3]}")

            positive_ranges = ranges[valid_ray]
            lidar_min_range_nonzero_m = (
                float(np.min(positive_ranges))
                if positive_ranges.size > 0
                else float("nan")
            )

            positive_ranges_xy = ranges_xy[valid_ray]
            lidar_min_range_xy_nonzero_m = (
                float(np.min(positive_ranges_xy))
                if positive_ranges_xy.size > 0
                else float("nan")
            )

            print("lidar shape:", diff_w.shape)
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
                contact = sensor.data.net_forces_w[0].detach().cpu().numpy()

                row = np.concatenate([
                    np.array([sim_step, sim_time_s], dtype=np.float64),
                    base_pos, base_quat, base_lin_vel, base_ang_vel,
                    joint_pos, joint_vel, torques, actions_np, commands_np, target,
                    np.array(
                        [
                            lidar_min_range_nonzero_m,
                            lidar_min_range_xy_nonzero_m,
                            float(
                                np.any(
                                    np.abs(contact[self._collision_link_indices, :])
                                    > self._collision_force_threshold
                                )
                            ),
                        ],
                        dtype=np.float64,
                    ),
                ])
                writer.writerow(row.tolist())


                # ===== CONTACT SAVE =====
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
                # (N_rays, 3): world-frame hit_w - sensor_origin_w; range = ||diff_w|| per ray.
                if sim_time_s + 1e-12 >= self.next_lidar_time_s:
                    np.save(
                        os.path.join(self.lidar_dir, f"lidar_{self.lidar_frame_idx:06d}.npy"),
                        diff_w,
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
