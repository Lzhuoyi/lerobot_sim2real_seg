import signal

# Script to visualize only segmentation data from ManiSkill environments
# Clusters segmentation IDs into 3 groups: Background (0,11,12), Arm (1-10,14), Cube (13)
from mani_skill.utils import common
from mani_skill.utils import visualization
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c

import argparse

import gymnasium as gym
import numpy as np
# Clustered color palette for 3 segmentation groups
# Background (0,11,12): Blue, Arm (1-10,14): Red, Cube (13): Green
cluster_colors = np.array([
    [100, 150, 255],  # Blue for Background
    [255, 100, 100],  # Red for Arm  
    [100, 255, 100],  # Green for Cube
], dtype=np.uint8)

def cluster_segmentation_ids(seg_ids):
    """Cluster segmentation IDs into 3 groups"""
    # Handle both 2D and 3D arrays
    if seg_ids.ndim == 3:
        seg_ids_flat = seg_ids.squeeze()  # Remove single dimension
    else:
        seg_ids_flat = seg_ids
    print("seg_ids:",seg_ids.ndim)
    
    clustered = np.zeros_like(seg_ids_flat)
    
    # Background cluster: IDs 0, 11, 12
    background_mask = np.isin(seg_ids_flat, [0, 11, 12])
    clustered[background_mask] = 0
    
    # Arm cluster: IDs 1-10, 14
    arm_mask = np.isin(seg_ids_flat, list(range(1, 11)) + [14])
    clustered[arm_mask] = 1
    
    # Cube cluster: ID 13
    cube_mask = (seg_ids_flat == 13)
    clustered[cube_mask] = 2
    
    return clustered
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera
from mani_skill.utils.structs import Actor, Link
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--id", type=str, help="The ID or name of actor you want to segment and render")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to run. Used for some basic testing and not visualized")
    parser.add_argument("--cam-width", type=int, help="Override the width of every camera in the environment")
    parser.add_argument("--cam-height", type=int, help="Override the height of every camera in the environment")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height

    env: BaseEnv = gym.make(
        args.env_id,
        # obs_mode="rgb+depth+segmentation",
        obs_mode="rgb+segmentation",
        num_envs=args.num_envs,
        sensor_configs=sensor_configs
    )

    obs, _ = env.reset(seed=args.seed)
    selected_id = args.id
    if selected_id is not None and selected_id.isdigit():
        selected_id = int(selected_id)

    n_cams = 0
    for config in env.unwrapped._sensors.values():
        if isinstance(config, Camera):
            n_cams += 1
    print(f"Visualizing {n_cams} segmentation cameras with 3 clusters:")
    print("  Cluster 0 (Blue): Background (IDs 0, 11, 12)")
    print("  Cluster 1 (Red): Arm (IDs 1-10, 14)")
    print("  Cluster 2 (Green): Cube (ID 13)")

    print("ID to Actor/Link name mappings")
    print("0: Background")

    reverse_seg_id_map = dict()
    for obj_id, obj in sorted(env.unwrapped.segmentation_id_map.items()):
        if isinstance(obj, Actor):
            print(f"{obj_id}: Actor, name - {obj.name}")
        elif isinstance(obj, Link):
            print(f"{obj_id}: Link, name - {obj.name}")
        reverse_seg_id_map[obj.name] = obj_id
    if selected_id is not None and not isinstance(selected_id, int):
        selected_id = reverse_seg_id_map[selected_id]

    renderer = visualization.ImageRenderer()
    
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        imgs = []
        for cam in obs["sensor_data"].keys():
            if "segmentation" in obs["sensor_data"][cam]:
                # "seg" is an array shaped (H, W, 1) is simply a single‑channel (grayscale or label) mask 
                # that matches the spatial dimensions of the input image and encodes exactly one label per pixel
                seg = common.to_numpy(obs["sensor_data"][cam]["segmentation"][0])
                if selected_id is not None:
                    seg = seg == selected_id # 2D array + extra "1" for channel dimension (H,W,1)
                
                # Cluster segmentation IDs and create RGB visualization
                clustered_seg = cluster_segmentation_ids(seg)
                seg_rgb = np.zeros((clustered_seg.shape[0], clustered_seg.shape[1], 3), dtype=np.uint8)
                
                # Apply clustered colors
                for cluster_id, color in enumerate(cluster_colors):
                    mask = clustered_seg == cluster_id
                    seg_rgb[mask] = color
                
                imgs.append(seg_rgb)
        
        # Tile images horizontally
        img = visualization.tile_images(imgs, nrows=1)
        renderer(img)

if __name__ == "__main__":
    main(parse_args())
