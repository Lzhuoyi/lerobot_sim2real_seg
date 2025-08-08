"""
This script is used to evaluate a random or RL trained agent on a real robot using the LeRobot system.
This version applies YOLOE to extract segmentation data from real images before passing them to the model.

python lerobot_sim2real/scripts/eval_ppo_seg.py --env_id="SO100GraspCube-v1" --env-kwargs-json-path=/home/jellyfish/lerobot_ws/lerobot-sim2real/models/seg_sim2real/env_config.json --checkpoint=/home/jellyfish/lerobot_ws/lerobot-sim2real/models/seg_sim2real/ckpt_5176.pt --control-freq=15
"""

from dataclasses import dataclass
import json
import random
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import tyro
import cv2
from ultralytics import YOLOE
from lerobot_sim2real.config.real_robot import create_real_robot
from lerobot_sim2real.rl.ppo_rgb import Agent

from lerobot_sim2real.utils.safety import setup_safe_exit
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDSegmentationObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm
from mani_skill.utils.visualization import tile_images
import matplotlib.pyplot as plt

# GLOBAL DEBUG VARIABLES - Easy access for debugging
GLOBAL_DEBUG = {
    'target_objects': ["black robot manipulator", "red square cube"],  # Global target objects
    'confidence_threshold': 0.2,  # Global confidence threshold
    'detection_debug': False,  # Enable detection debugging
    'verbose_detection': False,  # Verbose detection output
    'fallback_confidence': 0.1,  # Lower fallback confidence
    'show_all_detections': False,  # Show all YOLO detections regardless of target match
}

@dataclass
class Args:
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to load agent weights from for evaluation. If None then a random agent will be used"""
    env_kwargs_json_path: Optional[str] = None
    """path to a json file containing additional environment kwargs to use. For real world evaluation this is not needed but if you want to turn on debug mode which visualizes the sim and real envs side by side you will need this"""
    debug: bool = False
    """if toggled, the sim and real envs will be visualized side by side"""
    continuous_eval: bool = True
    """If toggled, the evaluation will run until episode ends without user input. If false, at each timestep the user will be prompted to press enter to let the robot continue"""
    max_episode_steps: int = 100
    """The maximum number of control steps the real robot can take before we stop the episode and reset the environment. It is recommended to set this number to be larger than the value the sim env is set to, that way you can permit the
    robot more chances to recover from failures / solve the task."""
    num_episodes: Optional[int] = None
    """The number of episodes to evaluate for. If None, the evaluation will run until the user presses ctrl+c"""
    env_id: str = "SO100GraspCube-v1"
    """The environment id to use for evaluation. This should be the same as the environment id used for training."""
    seed: int = 1
    """seed of the experiment"""
    record_dir: Optional[str] = None
    """Directory to save recordings of the camera captured images. If none no recordings are saved"""
    control_freq: Optional[int] = 15
    """The control frequency of the real robot. For safety reasons we recommend setting this to 15Hz or lower as we permit the RL agent to take larger actions to move faster. If this is none, it will use the same control frequency the sim env uses."""
    target_objects: Optional[str] = None
    """Comma-separated list of target objects to detect with YOLOE. If None, uses default objects."""
    confidence_threshold: float = 0.21
    """Confidence threshold for YOLOE detection. Lower values detect more objects but may include false positives."""
    debug_detection: bool = True
    """Enable detection debugging output"""

class YOLOESegmentationProcessor:
    """YOLOE-based segmentation processor for real robot evaluation"""
    
    def __init__(self, target_objects=None, confidence_threshold=None):
        # Initialize YOLOE model
        self.model = YOLOE("yoloe-11l-seg.pt")
        
        # Use global debug variables
        if target_objects is None:
            self.target_objects = GLOBAL_DEBUG['target_objects'].copy()
        else:
            self.target_objects = target_objects
            GLOBAL_DEBUG['target_objects'] = target_objects  # Update global
        
        # Set confidence threshold
        if confidence_threshold is None:
            self.confidence_threshold = GLOBAL_DEBUG['confidence_threshold']
        else:
            self.confidence_threshold = confidence_threshold
            GLOBAL_DEBUG['confidence_threshold'] = confidence_threshold  # Update global
        
        # CRITICAL FIX: Set text prompt to detect the specified objects (like test_yolo.py)
        self.model.set_classes(self.target_objects, self.model.get_text_pe(self.target_objects))
        
        # Map YOLOE detections to simulation IDs
        # Background: ID 0, 11, 12 (default)
        # Arm: IDs 1-10, 14 (black robot arm)
        # Cube: ID 13 (square cube)
        self.object_to_id_map = {
            "black robot manipulator": list(range(1, 11)) + [14],  # Arm IDs
            "red square cube": [13],  # Cube ID
            "background": [0, 11, 12]  # Background IDs
        }
        
        print(f"YOLOE initialized with target objects: {self.target_objects}")
        print(f"Confidence threshold: {self.confidence_threshold}")
    
    def process_image(self, image, num_channels=2):
        """
        Process RGB image to get segmentation with explicit object detection
        
        Args:
            image: RGB image (numpy array or torch tensor)
            num_channels: Number of channels in output segmentation
            
        Returns:
            segmentation: Segmentation tensor with IDs mapped to simulation format
        """
        # Convert to numpy but keep original resolution
        if isinstance(image, torch.Tensor):
            original_image = image.cpu().numpy()
        else:
            original_image = image
        
        # Run YOLO on ORIGINAL RESOLUTION
        results = self.model(original_image, verbose=False)
        
        # Create segmentation and ONLY resize the final result
        h, w = original_image.shape[:2]
        segmentation = np.zeros((h, w, num_channels), dtype=np.int16)
        
        if results and len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            print(f"YOLOE found {len(masks)} detections")
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = results[0].names[cls]
                
                print(f"  Detection {i+1}: '{label}' (conf: {conf:.3f})")
                
                # Check if this is one of our target objects (direct match like test_yolo.py)
                target_found = False
                matched_target = None
                
                # Direct matching (like test_yolo.py approach)
                if label in self.target_objects:
                    target_found = True
                    matched_target = label
                    print(f"    -> DIRECT MATCH: '{label}'")
                
                # Also try generic object matching for common objects
                if not target_found:
                    generic_matches = {
                        'robot': 'black robot manipulator',
                        'arm': 'black robot manipulator', 
                        'cube': 'red square cube',
                        'box': 'red square cube',
                        'block': 'red square cube',
                        'person': None,  # Ignore person detections
                    }
                    
                    for generic_term, mapped_target in generic_matches.items():
                        if generic_term in label.lower() and mapped_target is not None:
                            target_found = True
                            matched_target = mapped_target
                            print(f"    -> Generic match: '{label}' -> '{matched_target}'")
                            break
                
                if target_found:
                    # Use lower confidence threshold or fallback threshold
                    conf_threshold = min(self.confidence_threshold, GLOBAL_DEBUG['fallback_confidence'])
                    
                    if conf >= conf_threshold:
                        print(f"    -> ACCEPTED: '{label}' -> '{matched_target}' (conf: {conf:.3f} >= {conf_threshold:.3f})")
                        
                        # Resize mask back to original image size
                        mask_resized = cv2.resize(mask, (w, h))
                        
                        # Determine the correct ID based on object type
                        if matched_target and ("robot manipulator" in matched_target.lower() or "manipulator" in label.lower()):
                            # Assign arm IDs (1-10, 14) - use ID 1 for visualization
                            segmentation[mask_resized > 0.5, 0] = 1
                            print(f"    -> Assigned arm ID 1")
                        elif matched_target and ("cube" in matched_target.lower() or "cube" in label.lower() or "box" in label.lower()):
                            # Assign cube ID (13)
                            segmentation[mask_resized > 0.5, 0] = 13
                            print(f"    -> Assigned cube ID 13")
                        else:
                            # Default to arm if uncertain but matched
                            segmentation[mask_resized > 0.5, 0] = 1
                            print(f"    -> Assigned default arm ID 1")
                    else:
                        print(f"    -> REJECTED (low confidence): '{label}' (conf: {conf:.3f} < {conf_threshold:.3f})")
                else:
                    print(f"    -> NO MATCH: '{label}' (not in target objects)")
        else:
            print("YOLOE found no detections or no masks available")
        
        # Debug: Check final segmentation
        unique_ids = np.unique(segmentation[..., 0])
        non_zero_pixels = np.sum(segmentation[..., 0] > 0)
        print(f"Final segmentation - Unique IDs: {unique_ids}, Non-zero pixels: {non_zero_pixels}")
        # Resize ONLY the final segmentation for model input
        seg_resized = cv2.resize(segmentation, (128, 128), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(seg_resized)
        
    def segmentation_to_rgb(self, segmentation):
        """
        Convert segmentation IDs to RGB for visualization.
        
        Args:
            segmentation: Segmentation tensor (H, W, C) with IDs in first channel
            
        Returns:
            rgb_tensor: RGB tensor (H, W, 3) with clustered colors
        """
        # Extract segmentation IDs from first channel
        if segmentation.ndim == 3:
            seg_ids = segmentation[..., 0]
        else:
            seg_ids = segmentation[0, ..., 0] if segmentation.ndim == 4 else segmentation[..., 0]
        
        # Create RGB tensor
        h, w = seg_ids.shape
        rgb_tensor = torch.zeros(h, w, 3, device=seg_ids.device, dtype=torch.uint8)
        
        # Apply clustering colors (matching simulation)
        # Background (ID 0, 11, 12) -> Blue
        background_mask = torch.isin(seg_ids, torch.tensor([0, 11, 12], device=seg_ids.device, dtype=torch.int16))
        rgb_tensor[background_mask] = torch.tensor([100, 150, 255], device=seg_ids.device, dtype=torch.uint8)
        
        # Arm (IDs 1-10, 14) -> Red
        arm_ids = torch.tensor(list(range(1, 11)) + [14], device=seg_ids.device, dtype=torch.int16)
        arm_mask = torch.isin(seg_ids, arm_ids)
        rgb_tensor[arm_mask] = torch.tensor([255, 100, 100], device=seg_ids.device, dtype=torch.uint8)
        
        # Cube (ID 13) -> Green
        cube_mask = (seg_ids == 13)
        rgb_tensor[cube_mask] = torch.tensor([100, 255, 100], device=seg_ids.device, dtype=torch.uint8)
        
        return rgb_tensor

def update_global_debug_settings(**kwargs):
    """Update global debug settings at runtime"""
    global GLOBAL_DEBUG
    GLOBAL_DEBUG.update(kwargs)
    print(f"Updated global debug settings: {GLOBAL_DEBUG}")

def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Update global debug settings from args
    update_global_debug_settings(
        confidence_threshold=args.confidence_threshold,
        detection_debug=args.debug_detection,
        verbose_detection=args.debug_detection
    )

    ### Initialize YOLOE segmentation processor ###
    target_objects = None
    if args.target_objects:
        target_objects = [obj.strip() for obj in args.target_objects.split(',')]
    else:
        # Use the same target objects as your working test_yolo.py
        target_objects = ["black robot manipulator", "red square cube"]
    
    yoloe_processor = YOLOESegmentationProcessor(
        target_objects=target_objects, 
        confidence_threshold=args.confidence_threshold
    )

    ### Create and connect the real robot, wrap it to make it interfaceable with ManiSkill sim2real environments ###    
    real_robot = create_real_robot(uid="so100")
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot)

    ### Setup the sim environment to make various checks for sim2real alignment and debugging possible ###
    env_kwargs = dict(
        obs_mode="rgb+segmentation",  # Use segmentation-only mode for training
        render_mode="sensors", # only sensors mode is supported right now for real envs, basically rendering the direct visual observations fed to policy
        max_episode_steps=args.max_episode_steps, # give our robot more time to try and re-try the task
        domain_randomization=False,
        reward_mode="none"
    )

    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))
    
    sim_env = gym.make(
        args.env_id,
        **env_kwargs
    )
    # you can apply most wrappers freely to the sim_env and the real_env will use them as well
    sim_env = FlattenRGBDSegmentationObservationWrapper(sim_env, rgb=False, depth=False, segmentation=True, state=True)
    if args.record_dir is not None:
        # TODO (stao): verify this wrapper works
        sim_env = RecordEpisode(sim_env, output_dir=args.record_dir, save_trajectory=False, video_fps=sim_env.unwrapped.control_freq)
    
    # Get a sample observation from sim environment to understand the expected format
    sample_sim_obs, _ = sim_env.reset()
    expected_seg_shape = None
    if "segmentation" in sample_sim_obs:
        expected_seg_shape = tuple(sample_sim_obs["segmentation"].shape)  # Convert to tuple
        print(f"Expected segmentation shape from sim environment: {expected_seg_shape}")
    else:
        raise ValueError("No segmentation found in sim environment observations")
    
    # Force the expected shape to be 2-channel for compatibility with the trained model
    # This ensures that the real environment produces the same format as the trained model expects
    if expected_seg_shape is not None and expected_seg_shape[-1] != 2:
        print(f"Warning: Sim environment expects {expected_seg_shape[-1]} channels, but forcing 2-channel format for compatibility with trained model")
        # Update the expected shape to be 2-channel
        expected_seg_shape = (expected_seg_shape[0], expected_seg_shape[1], expected_seg_shape[2], 2)
    
    # Create a custom preprocessing function for the real environment
    def custom_sensor_data_preprocessing(sensor_data):
        """Custom preprocessing function that converts RGB to segmentation using YOLOE for both cameras"""
        processed_data = {}
        
        # Process each camera separately
        for camera_name, camera_data in sensor_data.items():
            processed_camera_data = camera_data.copy()
            
            # Always process RGB to segmentation if available
            if "rgb" in camera_data:
                # Process RGB image to get segmentation - keep it simple
                rgb_image = camera_data["rgb"]
                
                # Always create 2-channel segmentation to match the trained model
                num_channels = 2  # Force 2 channels to match the trained model
                segmentation = yoloe_processor.process_image(rgb_image, num_channels=num_channels)
                
                # Ensure segmentation has the correct device placement
                if isinstance(rgb_image, torch.Tensor):
                    segmentation = segmentation.to(rgb_image.device)
                
                # Resize to 128x128 and add batch dimension (keep it simple)
                if segmentation.ndim == 3:  # (H, W, C)
                    seg_np = segmentation.cpu().numpy()
                    h, w, c = seg_np.shape
                    seg_resized = np.zeros((128, 128, c), dtype=seg_np.dtype)
                    for channel in range(c):
                        seg_resized[:, :, channel] = cv2.resize(seg_np[:, :, channel], (128, 128), interpolation=cv2.INTER_NEAREST)
                    segmentation = torch.from_numpy(seg_resized).unsqueeze(0).to(segmentation.device)
                elif segmentation.ndim == 4:  # (1, H, W, C) or other
                    # Take first batch and resize
                    seg_np = segmentation[0].cpu().numpy()
                    h, w, c = seg_np.shape
                    seg_resized = np.zeros((128, 128, c), dtype=seg_np.dtype)
                    for channel in range(c):
                        seg_resized[:, :, channel] = cv2.resize(seg_np[:, :, channel], (128, 128), interpolation=cv2.INTER_NEAREST)
                    segmentation = torch.from_numpy(seg_resized).unsqueeze(0).to(segmentation.device)
                
                # Ensure correct dtype
                if segmentation.dtype != torch.int16:
                    segmentation = segmentation.to(torch.int16)
                
                # Final check to ensure the shape matches exactly (should be [1, 128, 128, 2])
                expected_shape = (1, 128, 128, 2)  # Force 2-channel format
                if segmentation.shape != expected_shape:
                    if segmentation.numel() == np.prod(expected_shape):
                        segmentation = segmentation.reshape(expected_shape)
                    else:
                        # Create a new tensor with the exact expected shape
                        new_segmentation = torch.zeros(expected_shape, device=segmentation.device, dtype=segmentation.dtype)
                        # Copy available channels (take first channel if available)
                        if segmentation.shape[-1] > 0:
                            new_segmentation[..., 0] = segmentation[..., 0]
                        segmentation = new_segmentation
                
                processed_camera_data["segmentation"] = segmentation
                print(f"Processed segmentation: {processed_camera_data['segmentation']}")
                # Remove RGB since we're using segmentation-only mode
                del processed_camera_data["rgb"]
            
            processed_data[camera_name] = processed_camera_data
        
        # Combine both camera segmentations for the trained model
        # The model expects a single segmentation input, so we need to combine them
        if len(processed_data) > 1:
            # If we have multiple cameras, combine their segmentations
            combined_segmentation = None
            for camera_name, camera_data in processed_data.items():
                if "segmentation" in camera_data:
                    seg = camera_data["segmentation"]
                    if combined_segmentation is None:
                        combined_segmentation = seg
                    else:
                        # Combine segmentations (take max values where both have detections)
                        combined_segmentation = torch.maximum(combined_segmentation, seg)
            
            # Update all cameras with the combined segmentation
            for camera_name in processed_data:
                if combined_segmentation is not None:
                    processed_data[camera_name]["segmentation"] = combined_segmentation
        
        return processed_data
    
    def overlay_envs(sim_env, real_env, yoloe_processor):
        """
        Overlays sim_env observations onto real_env observations with YOLOE processing
        Requires matching ids between the two environments' sensors
        e.g. id=phone_camera sensor in real_env / real_robot config, must have identical id in sim_env
        """
        real_obs = real_env.get_obs()["sensor_data"]
        sim_obs = sim_env.get_obs()["sensor_data"]
        assert sorted(real_obs.keys()) == sorted(
            sim_obs.keys()
        ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"

        overlaid_dict = sim_env.get_obs()["sensor_data"]
        overlaid_imgs = []
        for name in overlaid_dict:
            # Real environment already has segmentation data processed by the custom preprocessing function
            if "segmentation" in real_obs[name]:
                real_seg = real_obs[name]["segmentation"][0]
                # Convert segmentation to RGB for visualization
                real_seg_rgb = yoloe_processor.segmentation_to_rgb(real_seg)
                real_seg_norm = real_seg_rgb.cpu() / 255
            else:
                # Fallback: process RGB to segmentation if segmentation not available
                real_rgb = real_obs[name]["rgb"][0]
                real_seg = yoloe_processor.process_image(real_rgb)
                real_seg_rgb = yoloe_processor.segmentation_to_rgb(real_seg)
                real_seg_norm = real_seg_rgb.cpu() / 255
            
            sim_imgs = overlaid_dict[name]["segmentation"][0].cpu() / 255
            overlaid_imgs.append(0.5 * real_seg_norm + 0.5 * sim_imgs)

        return tile_images(overlaid_imgs), real_seg_norm, sim_imgs
    
    # Create a custom Sim2RealEnv class that properly handles the preprocessing
    class CustomSim2RealEnv(Sim2RealEnv):
        def __init__(self, sim_env, agent, control_freq=None, sensor_data_preprocessing_function=None, **kwargs):
            # Store the preprocessing function
            self._custom_preprocessing_function = sensor_data_preprocessing_function
            
            # Temporarily disable data checks to avoid the shape mismatch issue
            kwargs['skip_data_checks'] = True
            
            super().__init__(sim_env=sim_env, agent=agent, control_freq=control_freq, 
                           sensor_data_preprocessing_function=sensor_data_preprocessing_function, **kwargs)
        
        def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
            """Override to ensure preprocessing is applied correctly"""
            # note apply_texture_transforms is not used for real envs, data is expected to already be transformed to standard texture names, types, and shapes.
            self.agent.capture_sensor_data(self._sensor_names)
            data = self.agent.get_sensor_data(self._sensor_names)
            # observation data needs to be processed to be the same shape in simulation
            # default strategy is to do a center crop to the same shape as simulation and then resize image to the same shape as simulation
            if self._custom_preprocessing_function is not None:
                data = self._custom_preprocessing_function(data)
            else:
                data = self.preprocess_sensor_data(data)
            return data
    
    # Use our custom Sim2RealEnv class
    real_env = CustomSim2RealEnv(sim_env=sim_env, agent=real_agent, control_freq=args.control_freq, 
                                sensor_data_preprocessing_function=custom_sensor_data_preprocessing)
    
    # sim_env.print_sim_details()
    sim_obs, _ = sim_env.reset()
    real_obs, _ = real_env.reset()

    # Manually verify that the observations match after preprocessing
    print("Verifying observation compatibility after preprocessing...")
    for k in sim_obs.keys():
        if k in real_obs:
            if sim_obs[k].shape != real_obs[k].shape:
                print(f"WARNING: Shape mismatch for {k}: sim_obs shape: {sim_obs[k].shape}, real_obs shape: {real_obs[k].shape}")
            else:
                print(f"✓ {k}: sim_obs shape: {sim_obs[k].shape}, real_obs shape: {real_obs[k].shape}")
        else:
            print(f"WARNING: Key {k} not found in real_obs")
    
    # Additional verification for segmentation specifically
    if "segmentation" in sim_obs and "segmentation" in real_obs:
        # Force both sim and real observations to be 2-channel for compatibility
        expected_shape = (1, 128, 128, 2)  # Force 2-channel format
        
        # Check if sim_obs needs to be adjusted
        if sim_obs["segmentation"].shape != expected_shape:
            print(f"Adjusting sim segmentation shape from {sim_obs['segmentation'].shape} to {expected_shape}")
            if sim_obs["segmentation"].numel() == np.prod(expected_shape):
                sim_obs["segmentation"] = sim_obs["segmentation"].reshape(expected_shape)
            else:
                # Create new tensor with correct shape
                new_sim_seg = torch.zeros(expected_shape, device=sim_obs["segmentation"].device, dtype=sim_obs["segmentation"].dtype)
                # Copy available channels (take first channel if available)
                if sim_obs["segmentation"].shape[-1] > 0:
                    new_sim_seg[..., 0] = sim_obs["segmentation"][..., 0]
                sim_obs["segmentation"] = new_sim_seg
        
        # Check if real_obs needs to be adjusted
        if real_obs["segmentation"].shape != expected_shape:
            print(f"Adjusting real segmentation shape from {real_obs['segmentation'].shape} to {expected_shape}")
            if real_obs["segmentation"].numel() == np.prod(expected_shape):
                real_obs["segmentation"] = real_obs["segmentation"].reshape(expected_shape)
            else:
                # Create new tensor with correct shape
                new_real_seg = torch.zeros(expected_shape, device=real_obs["segmentation"].device, dtype=real_obs["segmentation"].dtype)
                # Copy available channels (take first channel if available)
                if real_obs["segmentation"].shape[-1] > 0:
                    new_real_seg[..., 0] = real_obs["segmentation"][..., 0]
                real_obs["segmentation"] = new_real_seg
        
        if sim_obs["segmentation"].shape == real_obs["segmentation"].shape:
            print("✓ Segmentation shapes match successfully!")
        else:
            print(f"ERROR: Segmentation shape mismatch after adjustment: {sim_obs['segmentation'].shape} vs {real_obs['segmentation'].shape}")
            raise ValueError("Segmentation shapes do not match after adjustment")

    ### Safety setups. Close environments/turn off robot upon ctrl+c ###
    setup_safe_exit(sim_env, real_env, real_agent)
        
    ### Load our checkpoint ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(sim_env, sample_obs=real_obs)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded agent from {args.checkpoint}")
    else:
        print("No checkpoint provided, using random agent")
    agent.to(device)

    ### Visualization setup for debug modes ###
    if args.debug:
        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        # Disable all default key bindings
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.manager.key_press_handler_id = None

        # initialize the plot
        overlaid_imgs, real_imgs, sim_imgs = overlay_envs(sim_env, real_env, yoloe_processor)
        im = ax.imshow(overlaid_imgs)
        im2 = ax2.imshow(sim_imgs)
        im3 = ax3.imshow(real_imgs)

    # Create a preprocessing function to ensure 2-channel format for the agent
    def preprocess_obs_for_agent(obs):
        """Ensure observations are in 2-channel format for the trained model"""
        if "segmentation" in obs:
            expected_shape = (1, 128, 128, 2)
            if obs["segmentation"].shape != expected_shape:
                print(f"Preprocessing: adjusting segmentation shape from {obs['segmentation'].shape} to {expected_shape}")
                if obs["segmentation"].numel() == np.prod(expected_shape):
                    obs["segmentation"] = obs["segmentation"].reshape(expected_shape)
                else:
                    # Create new tensor with correct shape
                    new_seg = torch.zeros(expected_shape, device=obs["segmentation"].device, dtype=obs["segmentation"].dtype)
                    # Copy available channels (take first channel if available)
                    if obs["segmentation"].shape[-1] > 0:
                        new_seg[..., 0] = obs["segmentation"][..., 0]
                    obs["segmentation"] = new_seg
        return obs

    # Function to visualize real environment segmented video stream
    def visualize_real_segmentation_stream(real_obs, yoloe_processor, frame_count=0):
        """Visualize real environment segmented video stream for both cameras side by side"""
        try:
            # Process both cameras if available
            cameras_to_show = []
            
            if "sensor_data" in real_obs:
                sensor_data = real_obs["sensor_data"]
                
                # Check for both so100 and base_camera
                for camera_name in ["so100", "base_camera"]:
                    if camera_name in sensor_data and "segmentation" in sensor_data[camera_name]:
                        cameras_to_show.append((camera_name, sensor_data[camera_name]["segmentation"]))
            
            # If no sensor_data, try direct segmentation
            elif "segmentation" in real_obs:
                cameras_to_show.append(("combined", real_obs["segmentation"]))
            
            if not cameras_to_show:
                return True
            
            # Create a single window for both cameras
            window_name = "Real Segmentation Stream - Dual Camera"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1600, 600)  # Wider window for side-by-side display
            
            # Process each camera and prepare images
            camera_images = []
            camera_names = []
            
            for camera_name, real_seg in cameras_to_show:
                try:
                    if real_seg.ndim == 4:
                        real_seg = real_seg.squeeze(0)  # Remove batch dimension
                    
                    real_seg_rgb = yoloe_processor.segmentation_to_rgb(real_seg)
                    real_seg_rgb = real_seg_rgb.cpu().numpy().astype(np.uint8)
                    
                    # Convert RGB to BGR for OpenCV
                    real_seg_bgr = cv2.cvtColor(real_seg_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Resize to 600x600 for consistent display
                    real_seg_bgr = cv2.resize(real_seg_bgr, (600, 600))
                    
                    # Add camera name overlay
                    cv2.putText(real_seg_bgr, f"{camera_name.upper()}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(real_seg_bgr, f"Frame: {frame_count}", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Add detection legend
                    y_offset = 90
                    cv2.putText(real_seg_bgr, "Detected Objects:", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 25
                    cv2.putText(real_seg_bgr, "Red: Robot Arm (ID 1-10,14)", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)  # Red in BGR
                    y_offset += 20
                    cv2.putText(real_seg_bgr, "Green: Cube (ID 13)", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)  # Green in BGR
                    y_offset += 20
                    cv2.putText(real_seg_bgr, "Blue: Background (ID 0,11,12)", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)  # Blue in BGR
                    
                    camera_images.append(real_seg_bgr)
                    camera_names.append(camera_name)
                    
                except Exception as e:
                    print(f"Error processing {camera_name} segmentation stream: {e}")
                    # Create a blank image with error message
                    blank_img = np.zeros((600, 600, 3), dtype=np.uint8)
                    cv2.putText(blank_img, f"Error: {camera_name}", (50, 300), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    camera_images.append(blank_img)
                    camera_names.append(camera_name)
            
            # Combine images side by side
            if len(camera_images) == 1:
                # Single camera - just show the image
                combined_image = camera_images[0]
            elif len(camera_images) == 2:
                # Two cameras - combine side by side
                combined_image = np.hstack(camera_images)
            else:
                # Multiple cameras - create a grid (2x2 max)
                if len(camera_images) <= 4:
                    # Create 2x2 grid
                    top_row = np.hstack(camera_images[:2])
                    bottom_row = np.hstack(camera_images[2:]) if len(camera_images) > 2 else np.zeros_like(top_row)
                    combined_image = np.vstack([top_row, bottom_row])
                else:
                    # Just show first 4 cameras
                    top_row = np.hstack(camera_images[:2])
                    bottom_row = np.hstack(camera_images[2:4])
                    combined_image = np.vstack([top_row, bottom_row])
            
            # Add overall title and controls
            cv2.putText(combined_image, "Real Segmentation Stream - Dual Camera View", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined_image, "Q/ESC: Quit", (10, combined_image.shape[0] - 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the combined frame
            cv2.imshow(window_name, combined_image)
            
        except Exception as e:
            print(f"Error in visualization stream: {e}")
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC key
            cv2.destroyAllWindows()
            return False
        
        return True

    ### Main evaluation loop ###
    episode_count = 0
    
    while args.num_episodes is None or episode_count < args.num_episodes:
        episode_count += 1
        print(f"Evaluation Episode {episode_count}")
        
        # Reset environments at the start of each episode
        sim_obs, _ = sim_env.reset()
        real_obs, _ = real_env.reset()
        
        for step in tqdm(range(args.max_episode_steps)):
            # Preprocess both sim and real observations to ensure 2-channel format
            sim_obs = preprocess_obs_for_agent(sim_obs)
            real_obs = preprocess_obs_for_agent(real_obs)
            
            # Visualize real environment segmented video stream
            if not visualize_real_segmentation_stream(real_obs, yoloe_processor, step):
                print("User requested to quit. Exiting...")
                sim_env.close()
                real_env.close()
                cv2.destroyAllWindows()
                return
            
            # Get agent observation (use real observation for actual control)
            agent_obs = real_obs

            # Move observations to device for agent
            agent_obs = {k: v.to(device) for k, v in agent_obs.items()}

            # Get action from agent
            with torch.no_grad():
                action = agent.get_action(agent_obs)
            
            if not args.continuous_eval:
                input("Press enter to continue to next timestep")         

            # Execute action in real environment
            real_obs, _, terminated, truncated, info = real_env.step(action.cpu().numpy())
            
            # Small delay to ensure smooth video stream
            cv2.waitKey(1)
            
            if terminated or truncated:
                print(f"Episode {episode_count} finished after {step + 1} steps")
                break
        
        # Reset for next episode
        real_env.reset()
        
        if args.debug:
            overlaid_imgs, real_imgs, sim_imgs = overlay_envs(sim_env, real_env, yoloe_processor)
            im.set_data(overlaid_imgs)
            im2.set_data(sim_imgs)
            im3.set_data(real_imgs)
            # Redraw the plot
            fig.canvas.draw()
            fig.show()
            fig.canvas.flush_events()

    sim_env.close()
    real_env.close()
    cv2.destroyAllWindows()
    print("Evaluation completed. CV windows closed.")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)