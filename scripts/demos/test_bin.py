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
import isaaclab.sim as sim_utils
from pxr import UsdGeom, Gf, Sdf
import omni.usd

ISAACLAB_LEG_IDXS = torch.tensor([
    0, 3, 7, 11, 15, 19,
    1, 4, 8, 12, 16, 20
])

class G1TurningCollector:
    """Collect locomotion data with correct yaw-angle normalization."""

    def __init__(self, vx=0.5, vy=0.0, yaw_rate=0.0,
                 waypoint=(2.0, 1.0), img_res=(640, 480),
                 save_every=10, collect_data=True):
        TASK = "Isaac-Velocity-Flat-G1-v0"
        RL_LIBRARY = "rsl_rl"
        self.collect_data = collect_data
        self.waypoint = np.array(waypoint)

        # --- RL config & checkpoint ---
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)

        # --- Environment ---
        env_cfg = G1FlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 100000
        env_cfg.curriculum = None
        env_cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0) 

        # --- Add Obstacle if you want ---
        self._add_obstacle_cube(env_cfg, pos=(0.0, 3.0, 0.25), size=(0.5, 1.0, 0.5),index=0)
        # self._add_blue_bin(env_cfg, pos=(2, 0, 0.25),index=0)
        #self._add_table(env_cfg, pos=(2, 2, 0.25),index=0)


        # --- Add Obstacle if you want ---
        
        
        
        # --- Add camera ---
        env_cfg.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_link/front_camera",
            update_period=0.05,
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
            update_period=0.05,

            offset=RayCasterCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
            ),

            mesh_prim_paths=["/World"],   

            ray_alignment="yaw",

            pattern_cfg=patterns.LidarPatternCfg(
                channels=32,                      # 垂直线数（先别太大）
                vertical_fov_range=(-90, 90),
                horizontal_fov_range=(-180, 180),
                horizontal_res=2.0,              # 分辨率（deg）
            ),

            debug_vis=False,
        )

        # --- Create environment ---
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        # load custom scene
        self._load_scene_usd()

        # --- Apply collider preset to USD-based objects ---
        self._apply_collider_to_blue_bin(index=0)

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

        # --- Output dirs ---
        self.base_dir = os.path.join(os.path.dirname(__file__), "../../data")
        self.image_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "g1_turning_yawfix_dataset.csv")

        # --- Robot info ---
        robot = self.env.unwrapped.scene["robot"]
        self.num_joints = robot.data.joint_pos.shape[1]
        print(f"[INFO] Detected {self.num_joints} actuated joints. Waypoint = {self.waypoint}")

        # --- Camera handle ---
        self.camera = self.env.unwrapped.scene["camera"]
        print(f"[INFO] Camera initialized. Data collection = {self.collect_data}")

         # --- lidar handle ---
        self.lidar_dir = os.path.join(self.base_dir, "lidar")
        os.makedirs(self.lidar_dir, exist_ok=True)



        # --- Add waypoint marker (green sphere) ---
        self._add_waypoint_marker()

        # --- Add obstacle cube ---
        #self._add_obstacle_cube(pos=(2.0, 0.0, 0.5), size=1.0)

    def _load_scene_usd(self):
        stage = omni.usd.get_context().get_stage()

        scene_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scene_new/lab_with_bin.usda")
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
        print(xform.GetLocalTransformation())
        # -----------------------------------------

        print("[INFO] Scene loaded")

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
                        kinematic_enabled=True
                    ),

                    collision_props=sim_utils.CollisionPropertiesCfg(),

                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=0.8,
                        restitution=0.0
                    ),

                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)
                    ),
                ),

                init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
            )
        )

        print(f"[INFO] Added {name} at {pos}")

    def _add_blue_bin(self, env_cfg, pos, index):

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg

        name = f"blue_bin_{index}"

        usd_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scene_new/blue_bin1.usdz")
        )

        setattr(
            env_cfg.scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",

                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,

                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True   
                    ),

                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True
                    ),

                    scale=(0.75, 0.75, 0.75),
                ),

                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos,
                    rot = (0.707, 0, 0, 0.707)  # quaternion (w,x,y,z)
                ),
            )
        )

        print(f"[INFO] Added blue bin at {pos}")

    def _apply_collider_to_blue_bin(self, index=0):
        """Apply CollisionAPI + MeshCollisionAPI to all Mesh prims inside the blue bin.

        UsdFileCfg only calls modify_collision_properties (not define_), so if the
        source USD has no CollisionAPI baked in, no collider is created.  This method
        does what the Isaac Sim "Collider Preset" button does: walk every Mesh prim
        and stamp the required physics schemas onto it.
        """
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

        stage = omni.usd.get_context().get_stage()
        bin_prim_path = f"/World/envs/env_0/blue_bin_{index}"
        bin_prim = stage.GetPrimAtPath(bin_prim_path)

        if not bin_prim.IsValid():
            print(f"[WARN] Blue bin prim not found at {bin_prim_path}")
            return

        for prim in Usd.PrimRange(bin_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue

            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)

            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_col.GetApproximationAttr().Set("convexDecomposition")

            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)

            print(f"  [INFO] Collider applied → {prim.GetPath()}")

        print(f"[INFO] Collision APIs applied to blue_bin_{index}")


    def _add_table(self, env_cfg, pos, index):

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg

        name = f"table_{index}"

        usd_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scene_new/table1.usdz")
        )

        setattr(
            env_cfg.scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",

                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,

                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True   
                    ),

                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True
                    ),


                    scale=(1, 1, 1),
                ),

                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos,
                    rot = (1, 0, 0, 0)  # quaternion (w,x,y,z)
                ),
            )
        )

        print(f"[INFO] Added table at {pos}")



    def _add_waypoint_marker(self):
        """Add green sphere markers for all waypoints."""
        stage = self.env.unwrapped.scene.stage

        
        if isinstance(self.waypoint[0], (float, int)):
            waypoints = [self.waypoint]
        else:
            waypoints = self.waypoint

        for i, wp in enumerate(waypoints):
            sphere_path = Sdf.Path(f"/World/WaypointMarker_{i}")
            if stage.GetPrimAtPath(sphere_path):
                continue 
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.GetRadiusAttr().Set(0.1)
            sphere.AddTranslateOp().Set(Gf.Vec3f(wp[0], wp[1], 0.0))
            sphere.CreateVisibilityAttr().Set("invisible")

            #color_attr = sphere.CreateDisplayColorAttr()
            #color_attr.Set([(0.0, 1.0, 0.0)])  # 绿色
        print(f"[INFO] Added {len(waypoints)} waypoint markers.")


    def quat_to_yaw(self, quat):
        w, x, y, z = quat  
        #print(f"x:{x},y:{y},z:{z},w:{w}")
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

        root_pose = torch.tensor([[0.0, 0.0, 0.65, 1.0, 0.0, 0.0, 0.0]], device=self.device)
        robot.write_root_pose_to_sim(root_pose)

        # support multi waypoints
        if isinstance(self.waypoint[0], (float, int)):
            waypoints = [self.waypoint]
        else:
            waypoints = self.waypoint

        print(f"[INFO] Running {num_steps} steps through {len(waypoints)} waypoints: {waypoints}")

        stop_thresh = 0.25
        k_yaw = 1.0
        max_yaw_rate = 1.0
        yaw_smooth = 0.1
        prev_yaw_rate = 0.0
        current_target_idx = 0
        threshold_deg = 55
        target = np.array(waypoints[current_target_idx])

        prev_yaw = 0.0
        prev_theta_v = 0.0

        # --- Data collection ---
        if self.collect_data:
            f = open(self.save_path, mode="w", newline="")
            writer = csv.writer(f)
            N = self.num_joints
            header = (
                [f"base_pos_{i}" for i in range(3)] +
                [f"base_quat_{i}" for i in range(4)] +
                [f"base_lin_vel_{i}" for i in range(3)] +
                [f"base_ang_vel_{i}" for i in range(3)] +
                [f"joint_pos_{i}" for i in range(N)] +
                [f"joint_vel_{i}" for i in range(N)] +
                [f"torque_{i}" for i in range(N)] +
                [f"action_{i}" for i in range(N)] +
                [f"command_{i}" for i in range(3)] +
                ["target_x", "target_y"]
            )
            writer.writerow(header)
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
                print(f"[INFO] Reached waypoint {current_target_idx+1}/{len(waypoints)} at step {step}, dist={dist:.3f}")
                current_target_idx += 1
                if current_target_idx >= len(waypoints):
                    print("[INFO] All waypoints reached. Stopping.")
                    break
                else:
                    target = np.array(waypoints[current_target_idx])
                    print(f"[INFO] Switching to next waypoint: {target}")
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

            print(f"theta_v: {np.degrees(theta_v)}")

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
            alpha = 0.2   # smoothing factor 0.1~0.3
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
            #action_new = torch.zeros_like(actions)
            #for i,new_i in enumerate(idx12):
                #action_new[:, new_i] = actions[:, new_i]   # 保留这12维
            #print(f"action_new: {action_new}")
            
            obs, _, _, _ = self.env.step(actions)

            # =========================
            # LIDAR COLLECTION (正确位置)
            # =========================
            lidar = self.env.unwrapped.scene["lidar"]

            lidar_points = lidar.data.ray_hits_w[0]   # (N_rays, 3)
            lidar_np = lidar_points.detach().cpu().numpy()

            # 防 nan
            lidar_np = np.nan_to_num(lidar_np, nan=0.0, posinf=0.0, neginf=0.0)

            # 转 range
            origin = base_pos
            ranges = np.linalg.norm(lidar_np - origin, axis=1)

            # debug 
            print("lidar shape:", lidar_np.shape)
            print("lidar min/max:", ranges.min(), ranges.max())
                

            # debug
            print(f"[STEP {step}] Target={target}, dist={dist:.2f}, yaw={np.degrees(yaw):.1f}°, "
                f"vx={vx_cmd:.2f}, vy={vy_cmd:.2f}, yaw_rate={np.degrees(yaw_rate):.1f}°/s")

            # save data
            if self.collect_data:
                base_lin_vel = data.root_lin_vel_w[0].cpu().numpy()
                base_ang_vel = data.root_ang_vel_w[0].cpu().numpy()
                joint_pos = data.joint_pos[0, :self.num_joints].cpu().numpy()
                joint_vel = data.joint_vel[0, :self.num_joints].cpu().numpy()
                torques = data.applied_torque[0, :self.num_joints].cpu().numpy()
                actions_np = actions[0,:self.num_joints].detach().cpu().numpy()
                commands_np = self.commands[0].detach().cpu().numpy()

                row = np.concatenate([
                    base_pos, base_quat, base_lin_vel, base_ang_vel,
                    joint_pos, joint_vel, torques, actions_np, commands_np, target
                ])
                writer.writerow(row.tolist())

                if step % self.save_every == 0:
                    # -------- Camera SAVE --------
                    rgb_tensor = self.camera.data.output["rgb"][0]

                    rgb_np = rgb_tensor[..., :3].cpu().numpy()

                    # if float
                    if rgb_np.dtype != np.uint8:
                        rgb_np = (rgb_np * 255).clip(0,255).astype(np.uint8)

                    # rotate 90°
                    rgb_np = np.rot90(rgb_np, k=1)

                    import imageio
                    imageio.imwrite(os.path.join(self.image_dir, f"rgb_{step:06d}.png"), rgb_np)

                    print(f"[DEBUG] Saved frame {step}")

                    # -------- LIDAR SAVE --------
                    lidar_dir = os.path.join(self.base_dir, "lidar")
                    os.makedirs(lidar_dir, exist_ok=True)

                    np.save(
                        os.path.join(lidar_dir, f"lidar_{step:06d}.npy"),
                        ranges   
                    )
        
        if self.collect_data and writer is not None:
            f.close()
            print(f"[INFO] Dataset saved to: {os.path.abspath(self.save_path)}")
            print(f"[INFO] Images written to: {os.path.abspath(self.image_dir)}")


def main():
    collect_flag = not args_cli.no_collect

    collector = G1TurningCollector(
    vx=args_cli.vx,
    vy=args_cli.vy,
    yaw_rate=args_cli.yaw_rate,
    waypoint=[(0,0),(8, 0.0)], 
    img_res=(640, 480),
    save_every=1,
    collect_data=collect_flag,
    )
    
       
    collector.run(num_steps=60000)
    simulation_app.close()


if __name__ == "__main__":
    main()
