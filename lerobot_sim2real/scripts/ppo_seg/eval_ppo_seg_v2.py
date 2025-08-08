"""
This script is used to evaluate a trained agent on a real robot using the LeRobot system.
Version 2: Clean structure with separate YOLO streaming and robot control loops.

python lerobot_sim2real/scripts/eval_ppo_seg_v2.py --env_id="SO100GraspCube-v1" --env-kwargs-json-path=/home/jellyfish/lerobot_ws/lerobot-sim2real/models/seg_sim2real/env_config.json --checkpoint=/home/jellyfish/lerobot_ws/lerobot-sim2real/models/seg_sim2real/ckpt_5176.pt --control-freq=15
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
import threading
import time

# Global variables for communication between threads
GLOBAL_SEGMENTATION_TENSOR = None
GLOBAL_SEGMENTATION_LOCK = threading.Lock()
GLOBAL_STOP_STREAMING = False

@dataclass
class Args:
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to load agent weights from for evaluation"""
    env_kwargs_json_path: Optional[str] = None
    """path to a json file containing additional environment kwargs to use"""
    debug: bool = False
    """if toggled, the sim and real envs will be visualized side by side"""
    continuous_eval: bool = True
    """If toggled, the evaluation will run until episode ends without user input"""
    max_episode_steps: int = 100
    """The maximum number of control steps the real robot can take"""
    num_episodes: Optional[int] = None
    """The number of episodes to evaluate for"""
    env_id: str = "SO100GraspCube-v1"
    """The environment id to use for evaluation"""
    seed: int = 1
    """seed of the experiment"""
    record_dir: Optional[str] = None
    """Directory to save recordings of the camera captured images"""
    control_freq: Optional[int] = 15
    """The control frequency of the real robot"""
    target_objects: Optional[str] = None
    """Comma-separated list of target objects to detect with YOLOE"""
    confidence_threshold: float = 0.25
    """Confidence threshold for YOLOE detection"""
    camera_indices: str = "4,2"
    """Comma-separated camera indices to use (e.g., '0,1' for cameras 0 and 1)"""

class YOLOSegmentationProcessor:
    """YOLO-based segmentation processor for real-time streaming"""
    
    def __init__(self, target_objects=None, confidence_threshold=0.25):
        # Initialize YOLO model
        self.model = YOLOE("yoloe-11l-seg.pt")
        
        # Set target objects
        if target_objects is None:
            self.target_objects = ["black robot manipulator", "red square cube"]
        else:
            self.target_objects = target_objects
        
        self.confidence_threshold = confidence_threshold
        
        # Set text prompt to detect the specified objects
        self.model.set_classes(self.target_objects, self.model.get_text_pe(self.target_objects))
        
        print(f"YOLO initialized with target objects: {self.target_objects}")
        print(f"Confidence threshold: {self.confidence_threshold}")
    
    def process_frame(self, frame):
        """
        Process a single frame to get segmentation
        
        Args:
            frame: RGB image (numpy array)
            
        Returns:
            segmentation: Segmentation tensor with clustered IDs
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Initialize segmentation tensor with background (ID 0)
        segmentation = np.zeros((h, w, 2), dtype=np.int16)
        
        if results and len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = results[0].names[cls]
                
                # Check if this is one of our target objects
                target_found = False
                matched_target = None
                
                # Direct matching
                if label in self.target_objects:
                    target_found = True
                    matched_target = label
                
                # Generic object matching
                if not target_found:
                    generic_matches = {
                        'robot': 'black robot manipulator',
                        'arm': 'black robot manipulator', 
                        'cube': 'red square cube',
                        'box': 'red square cube',
                        'block': 'red square cube',
                    }
                    
                    for generic_term, mapped_target in generic_matches.items():
                        if generic_term in label.lower() and mapped_target is not None:
                            target_found = True
                            matched_target = mapped_target
                            break
                
                if target_found and conf >= self.confidence_threshold:
                    # Resize mask back to original frame size
                    mask_resized = cv2.resize(mask, (w, h))
                    
                    # Determine the correct ID based on object type
                    if matched_target and ("robot manipulator" in matched_target.lower() or "manipulator" in label.lower()):
                        # Assign arm IDs (1-10, 14) - use ID 1 for visualization
                        segmentation[mask_resized > 0.5, 0] = 1
                    elif matched_target and ("cube" in matched_target.lower() or "cube" in label.lower() or "box" in label.lower()):
                        # Assign cube ID (13)
                        segmentation[mask_resized > 0.5, 0] = 13
                    else:
                        # Default to arm if uncertain but matched
                        segmentation[mask_resized > 0.5, 0] = 1
        
        return torch.from_numpy(segmentation)
    
    def cluster_segmentation(self, segmentation):
        """
        Cluster segmentation IDs into 3 groups: Background (0,11,12), Arm (1-10,14), Cube (13)
        Similar to demo_SO100 approach
        """
        if segmentation.ndim == 4:
            seg_ids = segmentation[0, ..., 0]  # Remove batch and use first channel
        elif segmentation.ndim == 3:
            seg_ids = segmentation[..., 0]  # Use first channel
        else:
            seg_ids = segmentation
        
        # Convert to int32 for consistent dtype handling
        seg_ids = seg_ids.int()
        clustered = torch.zeros_like(seg_ids, dtype=torch.int32)
        
        # Background cluster: IDs 0, 11, 12
        background_mask = torch.isin(seg_ids, torch.tensor([0, 11, 12], device=seg_ids.device, dtype=torch.int32))
        clustered[background_mask] = 0
        
        # Arm cluster: IDs 1-10, 14
        arm_ids = torch.tensor(list(range(1, 11)) + [14], device=seg_ids.device, dtype=torch.int32)
        arm_mask = torch.isin(seg_ids, arm_ids)
        clustered[arm_mask] = 1
        
        # Cube cluster: ID 13
        cube_mask = (seg_ids == 13)
        clustered[cube_mask] = 2
        
        return clustered
    
    def segmentation_to_rgb(self, segmentation):
        """
        Convert clustered segmentation to RGB for visualization
        """
        clustered = self.cluster_segmentation(segmentation)
        
        # Create RGB tensor
        h, w = clustered.shape
        rgb_tensor = torch.zeros(h, w, 3, device=clustered.device, dtype=torch.uint8)
        
        # Apply clustering colors (matching simulation)
        # Background (ID 0) -> Blue
        rgb_tensor[clustered == 0] = torch.tensor([100, 150, 255], device=clustered.device, dtype=torch.uint8)
        
        # Arm (ID 1) -> Red
        rgb_tensor[clustered == 1] = torch.tensor([255, 100, 100], device=clustered.device, dtype=torch.uint8)
        
        # Cube (ID 2) -> Green
        rgb_tensor[clustered == 2] = torch.tensor([100, 255, 100], device=clustered.device, dtype=torch.uint8)
        
        return rgb_tensor

def setup_cameras(camera_indices):
    """Setup cameras for streaming"""
    cameras = []
    indices = [int(idx.strip()) for idx in camera_indices.split(',')]
    
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cameras.append(cap)
            print(f"Camera {idx} opened successfully")
        else:
            print(f"Failed to open camera {idx}")
    
    return cameras

def yolo_streaming_loop(cameras, yoloe_processor):
    """
    High-frequency YOLO streaming loop
    This runs continuously and updates the global segmentation tensor
    """
    global GLOBAL_SEGMENTATION_TENSOR, GLOBAL_SEGMENTATION_LOCK, GLOBAL_STOP_STREAMING
    
    print("Starting YOLO streaming loop...")
    
    # Create window for visualization
    cv2.namedWindow("YOLO Segmentation Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Segmentation Stream", 1600, 600)
    
    frame_count = 0
    
    while not GLOBAL_STOP_STREAMING:
        try:
            # Process each camera
            camera_images = []
            
            for i, cap in enumerate(cameras):
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read from camera {i}")
                    continue
                
                # Process frame with YOLO
                segmentation = yoloe_processor.process_frame(frame)
                
                # Convert to RGB for visualization
                seg_rgb = yoloe_processor.segmentation_to_rgb(segmentation)
                seg_rgb = seg_rgb.cpu().numpy().astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
                
                # Resize to 600x600 for consistent display
                seg_bgr = cv2.resize(seg_bgr, (600, 600))
                
                # Add camera label
                cv2.putText(seg_bgr, f"Camera {i}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(seg_bgr, f"Frame: {frame_count}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                camera_images.append(seg_bgr)
            
            # Combine images side by side
            if len(camera_images) == 1:
                combined_image = camera_images[0]
            elif len(camera_images) == 2:
                combined_image = np.hstack(camera_images)
            else:
                # Create grid for multiple cameras
                top_row = np.hstack(camera_images[:2])
                bottom_row = np.hstack(camera_images[2:]) if len(camera_images) > 2 else np.zeros_like(top_row)
                combined_image = np.vstack([top_row, bottom_row])
            
            # Add overall title
            cv2.putText(combined_image, "YOLO Segmentation Stream - Real-time", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined_image, "Q/ESC: Quit", (10, combined_image.shape[0] - 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the combined frame
            cv2.imshow("YOLO Segmentation Stream", combined_image)
            
            # Update global segmentation tensor (use first camera's result)
            if camera_images:
                with GLOBAL_SEGMENTATION_LOCK:
                    # Store the processed segmentation for robot control
                    GLOBAL_SEGMENTATION_TENSOR = segmentation
                    if frame_count == 0:  # Debug first frame
                        print(f"YOLO: Updated global tensor with shape: {segmentation.shape}")
            else:
                if frame_count == 0:  # Debug first frame
                    print("YOLO: No camera images available")
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC key
                GLOBAL_STOP_STREAMING = True
                break
            
            frame_count += 1
            
        except Exception as e:
            print(f"Error in YOLO streaming loop: {e}")
            break
    
    # Cleanup
    for cap in cameras:
        cap.release()
    cv2.destroyAllWindows()
    print("YOLO streaming loop ended")

def transform_segmentation_for_model(segmentation_tensor):
    """
    Transform segmentation tensor to exact input format for CNN/model
    Reference to version 1 approach
    """
    if segmentation_tensor is None:
        # Return empty tensor if no segmentation available
        return torch.zeros((1, 128, 128, 2), dtype=torch.int16)
    
    # Ensure we have the right shape
    if segmentation_tensor.ndim == 3:
        # Add batch dimension
        segmentation_tensor = segmentation_tensor.unsqueeze(0)
    
    # Convert to numpy for processing
    seg_np = segmentation_tensor.cpu().numpy()
    
    # Handle different input shapes
    if seg_np.ndim == 4:
        h, w, c = seg_np.shape[1:]  # Remove batch dimension
    elif seg_np.ndim == 3:
        h, w, c = seg_np.shape
    else:
        # Fallback to default size
        h, w, c = 128, 128, 2
    
    seg_resized = np.zeros((128, 128, 2), dtype=seg_np.dtype)
    
    # Resize each channel
    for channel in range(min(c, 2)):  # Only use first 2 channels
        if seg_np.ndim == 4:
            channel_data = seg_np[0, :, :, channel]  # Remove batch dimension
        else:
            channel_data = seg_np[:, :, channel]
        seg_resized[:, :, channel] = cv2.resize(channel_data, (128, 128), interpolation=cv2.INTER_NEAREST)
    
    # Convert back to tensor with correct shape
    model_input = torch.from_numpy(seg_resized).unsqueeze(0).to(torch.int16)
    
    return model_input

def robot_control_loop(args, sim_env, real_env, agent, yoloe_processor):
    """
    Lower-frequency robot control loop
    Similar to eval_ppo_rgb.py structure
    """
    global GLOBAL_SEGMENTATION_TENSOR, GLOBAL_SEGMENTATION_LOCK, GLOBAL_STOP_STREAMING
    
    print("Starting robot control loop...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    
    episode_count = 0
    
    # Get initial observations (like eval_ppo_rgb.py)
    sim_obs, _ = sim_env.reset()
    real_obs, _ = real_env.reset()
    
    while args.num_episodes is None or episode_count < args.num_episodes:
        episode_count += 1
        print(f"Robot Control Episode {episode_count}")
        
        for step in tqdm(range(args.max_episode_steps)):
            # Get current segmentation from YOLO streaming
            with GLOBAL_SEGMENTATION_LOCK:
                current_segmentation = GLOBAL_SEGMENTATION_TENSOR
            
            # Transform segmentation for model input
            model_segmentation = transform_segmentation_for_model(current_segmentation)
            
            # Create observation for agent - use YOLO segmentation + state data
            agent_obs = {
                "segmentation": model_segmentation.to(device),
                "state": real_obs.get("state", torch.zeros((1, 0))).to(device)
            }
            
            # Get action from agent
            with torch.no_grad():
                action = agent.get_action(agent_obs)
            
            # Execute action in real environment
            real_obs, _, terminated, truncated, info = real_env.step(action.cpu().numpy())
            
            if not args.continuous_eval:
                input("Press enter to continue to next timestep")
            
            if terminated or truncated:
                print(f"Episode {episode_count} finished after {step + 1} steps")
                break
        
        # Reset for next episode (like eval_ppo_rgb.py)
        real_env.reset()
    
    print("Robot control loop ended")

def main(args: Args):
    """Main function with clean structure"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("LeRobot PPO Segmentation Evaluation - Version 2")
    print("="*60)
    
    # 1. Setup YOLO processor
    target_objects = None
    if args.target_objects:
        target_objects = [obj.strip() for obj in args.target_objects.split(',')]
    else:
        target_objects = ["black robot manipulator", "red square cube"]
    
    yoloe_processor = YOLOSegmentationProcessor(
        target_objects=target_objects, 
        confidence_threshold=args.confidence_threshold
    )
    
    # 2. Setup cameras for YOLO streaming
    cameras = setup_cameras(args.camera_indices)
    if not cameras:
        print("No cameras available. Exiting.")
        return
    
    # 3. Setup robot and environments (similar to eval_ppo_rgb.py)
    real_robot = create_real_robot(uid="so100")
    real_robot.connect()
    real_agent = LeRobotRealAgent(real_robot)
    
    # Setup sim environment - use segmentation mode but skip sensor data capture
    env_kwargs = dict(
        obs_mode="segmentation",
        render_mode="sensors",
        max_episode_steps=args.max_episode_steps,
        domain_randomization=False,
        reward_mode="none"
    )
    
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs.update(json.load(f))
    
    sim_env = gym.make(args.env_id, **env_kwargs)
    # Use segmentation wrapper but skip sensor data capture
    sim_env = FlattenRGBDSegmentationObservationWrapper(sim_env, rgb=False, depth=False, segmentation=True, state=True)
    
    if args.record_dir is not None:
        sim_env = RecordEpisode(sim_env, output_dir=args.record_dir, save_trajectory=False, video_fps=sim_env.unwrapped.control_freq)
    
    # Setup real environment - skip data checks since we're using direct camera access
    real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, control_freq=args.control_freq, skip_data_checks=True)
    
    # Get sample observations (like eval_ppo_rgb.py)
    sim_obs, _ = sim_env.reset()
    real_obs, _ = real_env.reset()
    
    # Safety setup
    setup_safe_exit(sim_env, real_env, real_agent)
    
    # Load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy segmentation observation for agent initialization
    # Use sim_obs structure as template since it has the correct format
    dummy_obs = sim_obs.copy()
    
    # Replace with dummy segmentation data
    if "segmentation" in dummy_obs:
        dummy_obs["segmentation"] = torch.zeros((1, 128, 128, 2), dtype=torch.int16)
    
    agent = Agent(sim_env, sample_obs=dummy_obs)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded agent from {args.checkpoint}")
    else:
        print("No checkpoint provided, using random agent")
    agent.to(device)
    
    # 4. Start YOLO streaming in separate thread
    yolo_thread = threading.Thread(
        target=yolo_streaming_loop, 
        args=(cameras, yoloe_processor)
    )
    # yolo_thread.daemon = True
    yolo_thread.start()
    
    # 5. Start robot control loop in main thread
    try:
        robot_control_loop(args, sim_env, real_env, agent, yoloe_processor)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop streaming
        GLOBAL_STOP_STREAMING = True
        yolo_thread.join(timeout=2)
        
        # Cleanup
        sim_env.close()
        real_env.close()
        cv2.destroyAllWindows()
        print("Evaluation completed.")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args) 