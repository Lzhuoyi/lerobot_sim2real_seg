import gymnasium as gym
import numpy as np
import sapien
import pygame
import time
import math
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple
import tyro
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode


@dataclass
class Args:
    """Command line arguments for the single arm control script"""
    env_id: str = "SO100GraspCube-v1"
    obs_mode: str = "none"
    robot_uids: Optional[str] = None
    sim_backend: str = "auto"
    reward_mode: Optional[str] = None
    num_envs: int = 1
    control_mode: Optional[str] = None
    render_mode: str = "human"
    shader: str = "default"
    record_dir: Optional[str] = None
    pause: bool = False
    quiet: bool = False
    seed: Optional[Union[int, List[int]]] = None



def setup_environment(args: Args) -> BaseEnv:
    """Set up the ManiSkill environment"""
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        if len(env_kwargs["robot_uids"]) == 1:
            env_kwargs["robot_uids"] = env_kwargs["robot_uids"][0]
    
    env = gym.make(args.env_id, **env_kwargs)
    
    if args.record_dir:
        record_dir = args.record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, 
                          save_trajectory=False, 
                          max_steps_per_video=gym_utils.find_max_episode_steps_value(env))
    
    return env


def main(args: Args):
    """Main control loop"""
    np.set_printoptions(suppress=True, precision=3)
    
    # Setup environment
    env = setup_environment(args)
    
    if not args.quiet:
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode:", env.unwrapped.control_mode)
        print("Reward mode:", env.unwrapped.reward_mode)
    
    # Initialize environment
    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])
    
    # Setup rendering
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    
    # Get robot instance
    robot = None
    if hasattr(env.unwrapped, "agent"):
        robot = env.unwrapped.agent.robot
    elif hasattr(env.unwrapped, "agents") and len(env.unwrapped.agents) > 0:
        robot = env.unwrapped.agents[0]
    
    # Main control loop
    clock = pygame.time.Clock()
    running = True
    
    while running:     
        action = env.action_space.sample()
        action = np.zeros_like(action)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render environment
        if args.render_mode is not None:
            env.render()
        # Check for episode termination
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
        
        clock.tick(60)  # Limit to 60 FPS
    
    # Cleanup
    env.close()
    
    if args.record_dir:
        print(f"Video saved to {args.record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)