#!/usr/bin/env python3
"""
Simulation keyboard teleoperation control for SO101 robot
Based on the real robot teleoperation script but adapted for ManiSkill simulation environment

Usage:
python sim_teleop_keyboard.py --env_id SO100GraspCube-v1 --control_mode pd_joint_pos
"""

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

import cv2
from ultralytics import YOLOE
import torch
# Define default object
DEFAULT_OBJECT = "black robotic manipulator"                # Default object to always detect
DEFAULT_OBJECT_COLOR = (244, 133, 66)            # Blue color for default object
DEFAULT_OBJECT_CONFIDENCE = 0.1                # Lower confidence threshold for default object

TARGET_OBJECT_COLOR = (55, 68, 219)              # Red color for target objects  
TARGET_OBJECT_CONFIDENCE = 0.4                   # Normal confidence threshold for target objects

BACKGROUND_COLOR = (168, 168, 168)               # Gray background color

# Global variable to track CV window creation
cv_window_created = False


@dataclass
class Args:
    """Command line arguments for the simulation control script"""
    env_id: str = "SO100GraspCube-v1"
    obs_mode: str = "rgb+segmentation"
    robot_uids: Optional[str] = None
    sim_backend: str = "auto"
    reward_mode: Optional[str] = None
    num_envs: int = 1
    control_mode: Optional[str] = "pd_joint_pos"  # Default to position control
    render_mode: str = "human"
    shader: str = "default"
    record_dir: Optional[str] = None
    pause: bool = False
    quiet: bool = False
    seed: Optional[Union[int, List[int]]] = None


class SO101SimKinematics:
    """Kinematics for SO101 robot simulation"""
    
    def __init__(self, l1=0.1159, l2=0.1350):
        self.l1 = l1  # Length of the first link (upper arm)
        self.l2 = l2  # Length of the second link (lower arm)
        
    def inverse_kinematics(self, x, y, l1=None, l2=None):
        """
        Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets
        
        Parameters:
            x: End effector x coordinate
            y: End effector y coordinate
            l1: Upper arm length (default uses instance value)
            l2: Lower arm length (default uses instance value)
            
        Returns:
            joint2_deg, joint3_deg: Joint angles in degrees (shoulder_lift, elbow_flex)
        """
        # Use instance values if not provided
        if l1 is None:
            l1 = self.l1
        if l2 is None:
            l2 = self.l2
            
        # Calculate joint2 and joint3 offsets in theta1 and theta2
        theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
        theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0
        
        # Calculate distance from origin to target point
        r = math.sqrt(x**2 + y**2)
        r_max = l1 + l2  # Maximum reachable distance
        
        # If target point is beyond maximum workspace, scale it to the boundary
        if r > r_max:
            scale_factor = r_max / r
            x *= scale_factor
            y *= scale_factor
            r = r_max
        
        # If target point is less than minimum workspace (|l1-l2|), scale it
        r_min = abs(l1 - l2)
        if r < r_min and r > 0:
            scale_factor = r_min / r
            x *= scale_factor
            y *= scale_factor
            r = r_min
        
        # Use law of cosines to calculate theta2
        cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
        
        # Clamp cos_theta2 to valid range [-1, 1] to avoid domain errors
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))
        
        # Calculate theta2 (elbow angle)
        theta2 = math.pi - math.acos(cos_theta2)
        
        # Calculate theta1 (shoulder angle)
        beta = math.atan2(y, x)
        gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
        theta1 = beta + gamma
        
        # Convert theta1 and theta2 to joint2 and joint3 angles
        joint2 = theta1 + theta1_offset
        joint3 = theta2 + theta2_offset
        
        # Ensure angles are within URDF limits
        joint2 = max(-0.1, min(3.45, joint2))
        joint3 = max(-0.2, min(math.pi, joint3))
        
        # Convert from radians to degrees
        joint2_deg = math.degrees(joint2)
        joint3_deg = math.degrees(joint3)
        # Apply coordinate system transformation
        joint2_deg = 90 - joint2_deg
        joint3_deg = joint3_deg - 90
        
        return joint2_deg, joint3_deg


class KeyboardController:
    """Keyboard controller with pygame display for simulation"""
    
    def __init__(self):
        pygame.init()
        pygame.font.init()
        
        # Window setup
        self.window_width = 800
        self.window_height = 600
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("SO101 Simulation Control")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        self.keys_pressed = set()
        self.controller_data = {}
        
    def get_action(self):
        """Get keyboard input and return action dictionary"""
        action = {}
        
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return {'quit': True}
            elif event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
            elif event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)
        
        # Map keys to actions
        key_mapping = {
            pygame.K_q: 'shoulder_pan_neg',
            pygame.K_a: 'shoulder_pan_pos',
            pygame.K_w: 'x_pos',
            pygame.K_s: 'x_neg',
            pygame.K_e: 'y_pos',
            pygame.K_d: 'y_neg',
            pygame.K_r: 'pitch_pos',
            pygame.K_f: 'pitch_neg',
            pygame.K_t: 'wrist_roll_neg',
            pygame.K_g: 'wrist_roll_pos',
            pygame.K_y: 'gripper_neg',
            pygame.K_h: 'gripper_pos',
            pygame.K_ESCAPE: 'quit',
            pygame.K_c: 'quit',
        }
        
        for key in self.keys_pressed:
            if key in key_mapping:
                action[key_mapping[key]] = True
                
        return action
    
    def update_controller_data(self, data):
        """Update controller data for display"""
        self.controller_data = data
    
    def draw_text(self, text, font, color, x, y):
        """Helper function to draw text"""
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
        return text_surface.get_height()
    
    def render(self):
        """Render the control interface"""
        self.screen.fill(self.BLACK)
        
        # Title
        y_pos = 10
        y_pos += self.draw_text("SO101 Simulation Keyboard Control", self.font_large, self.WHITE, 10, y_pos) + 10
        
        # Control instructions
        y_pos += self.draw_text("KEYBOARD CONTROLS:", self.font_medium, self.GREEN, 10, y_pos) + 5
        
        controls = [
            ("Joint Control:", self.YELLOW),
            ("  Q/A: Shoulder Pan -/+", self.WHITE),
            ("  T/G: Wrist Roll -/+", self.WHITE),
            ("  Y/H: Gripper -/+", self.WHITE),
            ("", self.WHITE),
            ("End-Effector Control:", self.YELLOW),
            ("  W/S: X coordinate +/-", self.WHITE),
            ("  E/D: Y coordinate +/-", self.WHITE),
            ("", self.WHITE),
            ("Orientation Control:", self.YELLOW),
            ("  R/F: Pitch adjustment +/-", self.WHITE),
            ("", self.WHITE),
            ("System:", self.YELLOW),
            ("  C/ESC: Quit program", self.RED),
        ]
        
        for text, color in controls:
            y_pos += self.draw_text(text, self.font_small, color, 10, y_pos) + 2
        
        # Status information
        status_x = 400
        status_y = 10
        
        status_y += self.draw_text("ROBOT STATUS:", self.font_medium, self.GREEN, status_x, status_y) + 5
        
        if self.controller_data:
            # Current position
            if 'position' in self.controller_data:
                pos = self.controller_data['position']
                status_y += self.draw_text(f"End-Effector Position:", self.font_small, self.YELLOW, status_x, status_y) + 2
                status_y += self.draw_text(f"  X: {pos[0]:.4f} m", self.font_small, self.WHITE, status_x, status_y) + 2
                status_y += self.draw_text(f"  Y: {pos[1]:.4f} m", self.font_small, self.WHITE, status_x, status_y) + 5
            
            # Joint angles
            if 'joint_angles' in self.controller_data:
                angles = self.controller_data['joint_angles']
                joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow Flex', 'Wrist Flex', 'Wrist Roll', 'Gripper']
                
                status_y += self.draw_text(f"Joint Angles (degrees):", self.font_small, self.YELLOW, status_x, status_y) + 2
                for i, (name, angle) in enumerate(zip(joint_names, angles)):
                    status_y += self.draw_text(f"  {name}: {angle:.1f}°", self.font_small, self.WHITE, status_x, status_y) + 2
                
                status_y += 5
            
            # Pitch adjustment
            if 'pitch' in self.controller_data:
                pitch = self.controller_data['pitch']
                status_y += self.draw_text(f"Pitch Adjustment: {pitch:.1f}°", self.font_small, self.YELLOW, status_x, status_y) + 5
            
            # Control frequency
            if 'frequency' in self.controller_data:
                freq = self.controller_data['frequency']
                status_y += self.draw_text(f"Control Frequency: {freq} Hz", self.font_small, self.BLUE, status_x, status_y) + 5
        
        # Currently pressed keys
        if self.keys_pressed:
            key_names = []
            key_name_map = {
                pygame.K_q: 'Q', pygame.K_a: 'A', pygame.K_w: 'W', pygame.K_s: 'S',
                pygame.K_e: 'E', pygame.K_d: 'D', pygame.K_r: 'R', pygame.K_f: 'F',
                pygame.K_t: 'T', pygame.K_g: 'G', pygame.K_y: 'Y', pygame.K_h: 'H',
                pygame.K_c: 'C', pygame.K_ESCAPE: 'ESC'
            }
            
            for key in self.keys_pressed:
                if key in key_name_map:
                    key_names.append(key_name_map[key])
            
            if key_names:
                status_y += self.draw_text(f"Pressed Keys: {', '.join(key_names)}", self.font_small, self.GREEN, status_x, status_y)
        
        pygame.display.flip()
    
    def cleanup(self):
        """Clean up pygame resources"""
        pygame.quit()


class SO101SimTeleopController:
    """SO101 simulation teleoperation controller"""
    
    def __init__(self, env, action_space):
        self.env = env
        self.action_space = action_space
        self.kinematics = SO101SimKinematics()
        
        # Control parameters
        self.control_freq = 50  # Control frequency(Hz)
        
        # End effector initial position
        self.current_x = 0.1629
        self.current_y = 0.1131
        
        # Pitch control parameter
        self.pitch = 0.0
        self.pitch_step = math.radians(1.0)  # Convert to radians
        
        # Current joint angles (in radians for simulation)
        self.current_joint_angles = np.zeros(6)  # [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        
        # Joint limits (in radians)
        self.joint_limits = {
            'shoulder_pan': (-math.pi, math.pi),
            'shoulder_lift': (-math.pi/2, math.pi/2),
            'elbow_flex': (-math.pi/2, math.pi/2),
            'wrist_flex': (-math.pi/2, math.pi/2),
            'wrist_roll': (-math.pi, math.pi),
            'gripper': (0, math.radians(45))
        }
        
        # Step sizes in radians
        self.joint_step = math.radians(5.0)  # 5 degrees converted to radians
        self.xy_step = 0.002  # meters
        
    def normalize_angle(self, angle, joint_idx):
        """Normalize angle from radians to [-1, 1] range"""
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        joint_name = joint_names[joint_idx]
        min_angle, max_angle = self.joint_limits[joint_name]
        return 2 * (angle - min_angle) / (max_angle - min_angle) - 1
    
    def denormalize_angle(self, normalized_value, joint_idx):
        """Denormalize value from [-1, 1] to radians"""
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        joint_name = joint_names[joint_idx]
        min_angle, max_angle = self.joint_limits[joint_name]
        return min_angle + (normalized_value + 1) * (max_angle - min_angle) / 2
    
    def handle_keyboard_input(self, keyboard_action):
        """
        Process keyboard input and update target positions
        
        Args:
            keyboard_action: Dictionary of keyboard actions
            
        Returns:
            bool: False if quit command detected, True otherwise
        """
        if not keyboard_action:
            return True
            
        # Check for quit command
        if 'quit' in keyboard_action:
            return False
            
        # Direct joint controls
        if 'shoulder_pan_pos' in keyboard_action:
            self.current_joint_angles[0] = min(self.current_joint_angles[0] + self.joint_step, 
                                             self.joint_limits['shoulder_pan'][1])
            
        if 'shoulder_pan_neg' in keyboard_action:
            self.current_joint_angles[0] = max(self.current_joint_angles[0] - self.joint_step,
                                             self.joint_limits['shoulder_pan'][0])
            
        if 'wrist_roll_pos' in keyboard_action:
            self.current_joint_angles[4] = min(self.current_joint_angles[4] + self.joint_step,
                                             self.joint_limits['wrist_roll'][1])
            
        if 'wrist_roll_neg' in keyboard_action:
            self.current_joint_angles[4] = max(self.current_joint_angles[4] - self.joint_step,
                                             self.joint_limits['wrist_roll'][0])
            
        if 'gripper_pos' in keyboard_action:
            self.current_joint_angles[5] = min(self.current_joint_angles[5] + self.joint_step,
                                             self.joint_limits['gripper'][1])
            
        if 'gripper_neg' in keyboard_action:
            self.current_joint_angles[5] = max(self.current_joint_angles[5] - self.joint_step,
                                             self.joint_limits['gripper'][0])
        
        # X-Y coordinate control
        if 'x_pos' in keyboard_action:
            self.current_x += self.xy_step
        if 'x_neg' in keyboard_action:
            self.current_x -= self.xy_step
        if 'y_pos' in keyboard_action:
            self.current_y += self.xy_step
        if 'y_neg' in keyboard_action:
            self.current_y -= self.xy_step
            
        # Calculate joint angles from X-Y coordinates using provided kinematics
        if any(key in keyboard_action for key in ['x_pos', 'x_neg', 'y_pos', 'y_neg']):
            shoulder_lift_deg, elbow_flex_deg = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            
            # Convert degrees to radians
            shoulder_lift_rad = math.radians(shoulder_lift_deg)
            elbow_flex_rad = math.radians(elbow_flex_deg)
            
            # Apply joint limits
            self.current_joint_angles[1] = np.clip(shoulder_lift_rad, 
                                                 self.joint_limits['shoulder_lift'][0],
                                                 self.joint_limits['shoulder_lift'][1])
            self.current_joint_angles[2] = np.clip(elbow_flex_rad,
                                                 self.joint_limits['elbow_flex'][0], 
                                                 self.joint_limits['elbow_flex'][1])
        
        # Pitch control
        if 'pitch_pos' in keyboard_action:
            self.pitch += self.pitch_step
        if 'pitch_neg' in keyboard_action:
            self.pitch -= self.pitch_step
        
        return True
    
    def update_wrist_flex_with_pitch(self):
        """Calculate wrist_flex based on shoulder_lift and elbow_flex with pitch adjustment"""
        wrist_flex = -self.current_joint_angles[1] - self.current_joint_angles[2] + self.pitch
        self.current_joint_angles[3] = np.clip(wrist_flex,
                                             self.joint_limits['wrist_flex'][0],
                                             self.joint_limits['wrist_flex'][1])
    
    def get_normalized_action(self):
        """Get current target positions normalized to [-1, 1]"""
        action = np.zeros(6)
        for i in range(6):
            action[i] = self.normalize_angle(self.current_joint_angles[i], i)
        return action
    
    def get_status_data(self):
        """Get current status data for display"""
        # Convert joint angles from radians to degrees for display
        joint_angles_deg = [math.degrees(angle) for angle in self.current_joint_angles]
        
        return {
            'position': (self.current_x, self.current_y),
            'joint_angles': joint_angles_deg,
            'pitch': math.degrees(self.pitch),
            'frequency': self.control_freq
        }
    
    def control_loop(self, keyboard, model, target_objects):
        """
        Main control loop
        
        Args:
            keyboard: Keyboard controller instance
        """

        print(f"Starting control loop, frequency: {self.control_freq}Hz")
        print("Check the pygame window for control instructions and robot status")
        print("="*50)
        
        running = True
        while running:
            try:
                # Update keyboard display
                keyboard.update_controller_data(self.get_status_data())
                keyboard.render()
                
                # Get keyboard input
                keyboard_action = keyboard.get_action()
                
                # Process keyboard input
                if not self.handle_keyboard_input(keyboard_action):
                    break  # Exit loop
                
                # Update wrist_flex position
                self.update_wrist_flex_with_pitch()
                
                # Get normalized action for simulation
                action = self.get_normalized_action()
                
                # Execute action in simulation
                obs, reward, terminated, truncated, info = self.env.step(action)
                # print("obs", obs["sensor_data"].keys())  # we will have "base_camera" and "so100", also "rgb" and "segmentation"
                print("base_camera", obs["sensor_data"]["base_camera"]["rgb"])
                # print("so100", obs["sensor_data"]["so100"]["segmentation"])

                # Run video stream loop and check if user wants to quit
                if not video_stream_loop(model, obs, target_objects):
                    break  # Exit loop if user pressed ESC

                # Render environment
                self.env.render()
                                
            except KeyboardInterrupt:
                print("User interrupted program")
                break
            except Exception as e:
                print(f"Error in control loop: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Cleanup
        keyboard.cleanup()


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
        robot_uids_list = args.robot_uids.split(",")
        if len(robot_uids_list) == 1:
            env_kwargs["robot_uids"] = robot_uids_list[0]
        else:
            env_kwargs["robot_uids"] = tuple(robot_uids_list)
    
    env = gym.make(args.env_id, **env_kwargs)
    
    if args.record_dir:
        record_dir = args.record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, 
                          save_trajectory=False, 
                          max_steps_per_video=gym_utils.find_max_episode_steps_value(env))
    
    return env

def video_stream_loop(model, obs, target_objects=None):
    global cv_window_created
    if target_objects is None:
        target_objects = [DEFAULT_OBJECT]
    """
    Independent video streaming loop that displays segmentation masks in a CV window
    Does not control the robot - purely for visual feedback
    
    Color scheme (configurable at top of file):
    - Default object: Blue - confidence threshold: DEFAULT_OBJECT_CONFIDENCE
    - Other target objects: Red - confidence threshold: TARGET_OBJECT_CONFIDENCE
    - Background: Gray
    """
    try:
        frame = obs["sensor_data"]["base_camera"]["rgb"]
        
        # Convert tensor to numpy and handle format
        if isinstance(frame, torch.Tensor):
            # Remove batch dimension if present and convert to numpy
            if frame.ndim == 4:
                frame = frame.squeeze(0)  # Remove batch dimension
            frame = frame.cpu().numpy()
        
        # Convert from RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize frame to 640x640 for YOLOE model
        frame_resized = cv2.resize(frame, (640, 640))
        
        # Convert back to RGB for YOLOE
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Run YOLOE inference
        results = model(frame_rgb)
        
        # Get original frame dimensions
        h, w = frame.shape[:2]
        
        # Create gray background
        annotated_frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        annotated_frame[:] = BACKGROUND_COLOR  # Set all pixels to gray
        
        if not results or not hasattr(results[0], 'masks') or not results[0].masks:
            # No objects detected - just show gray background
            pass
        else:
            # Process segmentation masks
            masks = results[0].masks.data.cpu().numpy()  # Get mask data
            boxes = results[0].boxes
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                cls = int(box.cls[0])
                label = results[0].names[cls]
                
                if label in target_objects:
                    # Resize mask from 640x640 back to original frame size
                    mask = cv2.resize(mask, (w, h))
                    
                    # Assign color based on object type and set confidence threshold
                    if label == DEFAULT_OBJECT:
                        color = DEFAULT_OBJECT_COLOR          # Color for default object
                        confidence_threshold = DEFAULT_OBJECT_CONFIDENCE  # Lower threshold for default object
                    else:
                        color = TARGET_OBJECT_COLOR           # Color for target objects
                        confidence_threshold = TARGET_OBJECT_CONFIDENCE   # Normal threshold for other objects
                    
                    # Apply color to mask area
                    mask_bool = mask > confidence_threshold  # Convert to boolean mask
                    annotated_frame[mask_bool] = color
        
        # Create named window only once (like in the working example)
        if not cv_window_created:
            try:
                cv2.namedWindow("YOLOE Segmentation", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLOE Segmentation", 640, 480)
                cv_window_created = True
                print("CV window created successfully")
            except Exception as e:
                print(f"Failed to create CV window: {e}")
                print("Continuing without display window...")
                return True
        
        # Display the annotated frame in the window
        try:
            cv2.imshow("YOLOE Segmentation", annotated_frame)
            
            # Handle key press (like in the working example)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC key
                cv2.destroyAllWindows()
                cv_window_created = False
                return False
        except Exception as e:
            print(f"Failed to display frame: {e}")
            return True

    except Exception as e:
        print(f"Video stream error: {e}")
        import traceback
        traceback.print_exc()
        try:
            cv2.destroyAllWindows()
            cv_window_created = False
        except:
            pass
    
    return True
    
    

def main(args: Args):
    """Main function"""
    np.set_printoptions(suppress=True, precision=3)
    
    print("LeRobot SO101 Simulation Keyboard Control")
    print("="*50)

    # Initialize YOLOE and camera
    model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes
    
    # Get detection targets from user input
    print("\n" + "="*60)
    print("YOLOE Detection Target Setup")
    print("="*60)
    print(f"Default: '{DEFAULT_OBJECT}' will always be detected (Blue color)")
    target_input = input("Enter additional objects to detect (separate multiple objects with commas, e.g., bottle,cup,mouse): ").strip()
    
    # Always include default object as the first target
    target_objects = [DEFAULT_OBJECT]
    
    # Add user-specified targets
    if target_input:
        # Parse multiple objects separated by commas
        additional_objects = [obj.strip() for obj in target_input.split(',') if obj.strip()]
        target_objects.extend(additional_objects)
        print(f"Detection targets: {target_objects}")
        print(f"Colors: '{DEFAULT_OBJECT}'=Blue, Others=Red")
    else:
        print(f"Detection targets: {target_objects} (only default object)")
        print(f"Colors: '{DEFAULT_OBJECT}'=Blue")
    
    # Set text prompt to detect the specified objects
    model.set_classes(target_objects, model.get_text_pe(target_objects))
    
    try:
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
            # Handle both int and list types for seed
            if isinstance(args.seed, list):
                seed_value = args.seed[0]
            else:
                seed_value = args.seed
            env.action_space.seed(seed_value)
        
        # Setup rendering
        if args.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = args.pause
            env.render()
        
        # Create keyboard controller
        keyboard = KeyboardController()
        
        # Create teleoperation controller
        controller = SO101SimTeleopController(env, env.action_space)
        
        print("Environment setup complete!")
        print(f"Initial end-effector position: x={controller.current_x:.4f}, y={controller.current_y:.4f}")
        print("A pygame window will open with control instructions and robot status.")
        
        # Start control loop
        controller.control_loop(keyboard, model, target_objects)
        
        # Cleanup
        env.close()
        
        if args.record_dir:
            print(f"Video saved to {args.record_dir}")
            
        print("Program ended")
        
    except Exception as e:
        print(f"Program execution failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please check:")
        print("1. Environment ID is correct")
        print("2. Control mode is supported")
        print("3. All dependencies are installed")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)