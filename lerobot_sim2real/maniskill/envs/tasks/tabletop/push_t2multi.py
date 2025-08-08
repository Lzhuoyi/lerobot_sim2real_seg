from typing import Any, Dict

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import FloatingPandaGripperFin
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose  # 添加这行
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


# extending TableSceneBuilder and only making 2 changes:
# 1.Making table smooth and white, 2. adding support for keyframes of new robots - panda stick
class WhiteTableSceneBuilder(TableSceneBuilder):
    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        b = len(env_idx)
        print(self.env.robot_uids, "robot_uids initialize")
        if self.env.robot_uids == "floating_panda_gripper_fin":
            print(self.env.robot_uids, "robot_uids initialize!!!!!!!!!!!")
            keyframe = self.env.agent.keyframes['open_facing_down']
            self.env.agent.reset(keyframe.qpos)  # 设置关节角度
            self.env.agent.robot.set_pose(keyframe.pose)  # 设置基座位置

    def build(self):
        super().build()
        # cheap way to un-texture table
        for part in self.table._objs:
            for triangle in (
                part.find_component_by_type(sapien.render.RenderBodyComponent)
                .render_shapes[0]
                .parts
            ):
                triangle.material.set_base_color(np.array([255, 255, 255, 255]) / 255)
                triangle.material.set_base_color_texture(None)
                triangle.material.set_normal_texture(None)
                triangle.material.set_emission_texture(None)
                triangle.material.set_transmission_texture(None)
                triangle.material.set_metallic_texture(None)
                triangle.material.set_roughness_texture(None)


@register_env("PushT-v2multi", max_episode_steps=100)
class PushT2MultiEnv(BaseEnv):
    """
    Task Description
    ----------------
    Easier Digital Twin of real life push-T task from Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
    "In this task, the robot needs to
    1 precisely push the T- shaped block into the target region, and
    2 move the end-effector to the end-zone which terminates the episode. [2 Not required for PushT-easy-v1]"

    Randomizations
    --------------
    - 3D T block initial position on table  [-1,1] x [-1,2] + T Goal initial position
    - 3D T block initial z rotation         [0,2pi]

    Success Conditions
    ------------------
    - The T block covers 90% of the 2D goal T's area

    Identical Parameters
    --------------------
    - 3D T block                     (3D cad link in their github README: https://github.com/real-stanford/diffusion_policy)
    - TODO (xhin): ur5e end-effector (3D cad link in their github README: https://github.com/real-stanford/diffusion_policy)

    Params To-Tune (Unspecified Real-World Parameters)
    --------------------------------------------------
    - Randomizations
    - T Goal initial position on table      [-0.156,-0.1] (center of mass of T)
    - T Goal initial z rotation             (5pi/3)
    - End-effector initial position         [-0.322, 0.284, 0.024]
    - intersection % threshold for success  90%
    - Table View Camera parameters          sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])

    TODO's (xhin):
    --------------
    - Add hand mounted camera for panda_stick robot, for visual rl
    - Add support for ur5e robot with hand mounted camera and real life end effector (3D cad link in their github README)
    - Tune Unspecified Real-World Parameters
    - Add robot qpos to randomizations
    """

    SUPPORTED_ROBOTS = ["floating_panda_gripper_fin"]
    agent: FloatingPandaGripperFin

    # 添加类属性来统一设置物体数量
    NUM_OBJECTS = 64 # 每个环境中的物体数量

    # # # # # # # # All Unspecified real-life Parameters Here # # # # # # # #
    # Randomizations
    # 3D T center of mass spawnbox dimensions
    tee_spawnbox_xlength = 0.2
    tee_spawnbox_ylength = 0.3

    # translation of the spawnbox from goal tee as upper left of spawnbox
    tee_spawnbox_xoffset = -0.06
    tee_spawnbox_yoffset = 0
    #  end randomizations - rotation around z is simply uniform

    # Hand crafted params to match visual of real life setup
    # T Goal initial position on table
    goal_offset = torch.tensor([-0.156, -0.1])
    goal_z_rot = (5 / 3) * np.pi
    goal_radius = 0.05  # radius of the goal T block

    # end effector goal - NOTE that chaning this will not change the actual
    # ee starting position of the robot - need to change joint position resting
    # keyframe in table setup to change ee starting location, then copy that location here
    ee_starting_pos2D = torch.tensor([0, 0, 0.5])
    # this will be used in the state observations
    ee_starting_pos3D = torch.tensor([0, 0, 0.5])

    # intersection threshold for success in T position
    intersection_thresh = 0.90

    # T block design choices
    T_mass = 0.8
    T_dynamic_friction = 0.3  # 从3改为0.3
    T_static_friction = 0.3   # 从3改为0.3

    def __init__(
        self, *args, robot_uids="floating_panda_gripper_fin", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.08, 0, 0.45], target=[-0.08, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=1.4,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[2, 0, 3.6], target=[-0.05, 0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=0.5, near=0.01, far=100
        )
    

    
    def _load_scene(self, options: dict):
        self.ee_starting_pos2D = self.ee_starting_pos2D.to(self.device)
        self.ee_starting_pos3D = self.ee_starting_pos3D.to(self.device)

        self.table_scene = WhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius*5,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0.11, 0.11, 1e-3]),
        )

        def create_te(name="te", base_color=np.array([194, 19, 22, 255]) / 255, r=0.07, n_min=3, n_max=10, seed=66, num_objs=128):
            # Set the seed if provided
            if seed is not None:
                torch.manual_seed(seed)
            
            # 创建多个物体，每个环境有num_objs个物体
            te_objects = []
            for i in range(self.num_envs):
                # 为每个环境创建num_objs个物体
                for obj_idx in range(num_objs):
                    builder = self.scene.create_actor_builder()
                    builder._mass = self.T_mass
                    te_material = sapien.pysapien.physx.PhysxMaterial(
                        static_friction=self.T_dynamic_friction,
                        dynamic_friction=self.T_static_friction,
                        restitution=0,
                    )

                    # 为每个物体单独生成随机顶点数
                    n_vertices = torch.randint(n_min, n_max + 1, (1,), device=self.device)[0].item()
                    
                    # 修改角度生成逻辑
                    if n_vertices < 5:
                        section_size = 2 * np.pi / n_vertices
                        angles = []
                        for j in range(n_vertices):
                            section_start = j * section_size
                            section_end = (j + 1) * section_size
                            angle = section_start + torch.rand(1, device=self.device).item() * (section_end - section_start)
                            angles.append(angle)
                        angles = np.array(angles)
                    else:
                        angles = torch.sort(torch.rand(n_vertices, device=self.device) * (2 * torch.pi))[0].cpu().numpy()
                    
                    # 计算顶点位置
                    vertices = []
                    for angle in angles:
                        x = r * np.cos(angle)
                        y = r * np.sin(angle)
                        vertices.append([x, y])
                    vertices = np.array(vertices)

                    # 为每条边创建盒子
                    half_thickness = 0.03
                    box_width = 0.02
                    
                    for j in range(n_vertices):
                        v1 = vertices[j]
                        v2 = vertices[(j + 1) % n_vertices]
                        
                        center = (v1 + v2) / 2
                        edge = v2 - v1
                        length = np.linalg.norm(edge)
                        angle = np.arctan2(edge[1], edge[0])
                        
                        box_pose = sapien.Pose(
                            p=np.array([center[0]*0.9, center[1]*0.9, 0.0]),
                            q=euler2quat(0, 0, angle)
                        )
                        
                        box_half_size = [length/2, box_width/2, half_thickness]
                        builder.add_box_collision(
                            pose=box_pose,
                            half_size=box_half_size,
                            material=te_material,
                        )

                        # Define base colors for interpolation
                        red_color = np.array([194, 19, 22, 255]) / 255
                        white_color = np.array([240, 160, 160, 255]) / 255
                        
                        # Random interpolation factor for each object
                        # t = torch.rand(1, device=self.device).item()
                        t = 0.0
                        object_color = red_color * (1 - t) + white_color * t

                        builder.add_box_visual(
                            pose=box_pose,
                            half_size=box_half_size,
                            material=sapien.render.RenderMaterial(
                                base_color=object_color,  # Use interpolated color
                            ),
                        )
                    
                    # 设置碰撞组
                    # 1是默认组（包括桌面），2^n是物体组
                    collision_group = 6  # 物体的碰撞组ID
                    collision_mask = 1 | collision_group   # 可以与默认组(1)和自己的组发生碰撞
                    # 设置四个参数：[collision_group, collision_mask, collision_filter, reserved]
                    builder.collision_groups = [collision_group, collision_mask, collision_group, 0]
                    
                    # 设置场景索引，确保每个物体属于正确的环境
                    builder.set_scene_idxs([i])
                    # 构建物体并添加到列表
                    te_objects.append(builder.build(name=f"{name}_{i}_{obj_idx}"))

            # 合并所有物体
            return Actor.merge(te_objects, name=name)

        # 使用类属性NUM_OBJECTS
        self.te = create_te(name="Te", num_objs=self.NUM_OBJECTS)

    def quat_to_z_euler(self, quats):
        assert len(quats.shape) == 2 and quats.shape[-1] == 4
        # z rotation == can be defined by just qw = cos(alpha/2), so alpha = 2*cos^{-1}(qw)
        # for fixing quaternion double covering
        # for some reason, torch.sign() had bugs???
        signs = torch.ones_like(quats[:, -1])
        signs[quats[:, -1] < 0] = -1.0
        qw = quats[:, 0] * signs
        z_euler = 2 * qw.acos()
        return z_euler

    def quat_to_zrot(self, quats):
        # expecting batch of quaternions (b,4)
        assert len(quats.shape) == 2 and quats.shape[-1] == 4
        # output is batch of rotation matrices (b,3,3)
        alphas = self.quat_to_z_euler(quats)
        # constructing rot matrix with rotation around z
        rot_mats = torch.zeros(quats.shape[0], 3, 3).to(quats.device)
        rot_mats[:, 2, 2] = 1
        rot_mats[:, 0, 0] = alphas.cos()
        rot_mats[:, 1, 1] = alphas.cos()
        rot_mats[:, 0, 1] = -alphas.sin()
        rot_mats[:, 1, 0] = alphas.sin()
        return rot_mats

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # 使用类属性NUM_OBJECTS
            # 为每个物体生成位置 (b*num_objs, 3)
            target_region_xyz = torch.zeros((b * self.NUM_OBJECTS, 3))
            for i in range(self.NUM_OBJECTS):
                start_idx = i * b
                end_idx = (i + 1) * b
                
                # 为每个物体设置稍微不同的初始位置范围
                offset_x = self.tee_spawnbox_xoffset   # 在x方向上错开
                offset_y = self.tee_spawnbox_yoffset
                
                target_region_xyz[start_idx:end_idx, 0] += (
                    torch.rand(b) * 0.01 + offset_x
                )
                target_region_xyz[start_idx:end_idx, 1] += (
                    torch.rand(b) * 0.01 + offset_y
                )
                target_region_xyz[start_idx:end_idx, 2] = 0.06 / 2 + 1e-3

            # 为每个物体生成朝向 (b*num_objs, 4)
            q_euler_angle = torch.rand(b * self.NUM_OBJECTS) * (2 * torch.pi)
            q = torch.zeros((b * self.NUM_OBJECTS, 4))
            q[:, 0] = (q_euler_angle / 2).cos()
            q[:, -1] = (q_euler_angle / 2).sin()

            # 设置所有物体的位姿
            obj_pose = Pose.create_from_pq(p=target_region_xyz, q=q)
            self.te.set_pose(obj_pose)

            # Define Goal Pose
            goal_region_xyz = torch.tensor([-0.121, 0.121, 0]) 
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=goal_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def _get_obs_extra(self, info: Dict):
        # ee position is super useful for pandastick robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            # state based gets info on goal position - necessary to learn task
            # Create target position tensor for all environments
            num_envs = len(self.agent.tcp.pose.p)
            target_pos = torch.tensor([-0.156, -0.1, 0.06], device=self.device).unsqueeze(0).expand(num_envs, -1)
            
            obs.update(
                goal_pos=target_pos,  # Target position for multi-object pushing
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # 获取每个环境中所有物体的平均位置
        num_envs = len(self.agent.tcp.pose.p)
        
        # 重塑物体位置为 [num_envs, num_objs, 3]
        obj_positions = self.te.pose.p.reshape(num_envs, self.NUM_OBJECTS, 3)
        
        # 计算每个环境中物体的平均位置 [num_envs, 3]
        mean_obj_positions = obj_positions.mean(dim=1)
        
        # 计算TCP到平均位置的距离
        tcp_to_push_pose = mean_obj_positions - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reward = (1 - torch.tanh(5 * tcp_to_push_pose_dist))

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 1.0  # Updated since max of (1-tanh(x)) is 1
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward













