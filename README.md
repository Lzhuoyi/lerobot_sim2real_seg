# LeRobot Sim2Real-seg
This is a modification based on Stone's Lerobot rgb sim2real minimal framework


## Getting Started

Install this repo by running the following
```bash
conda create -n ms3-lerobot "python==3.11" # 3.11 is recommended
git clone https://github.com/StoneT2000/lerobot-sim2real.git
pip install -e .
pip install torch # install the version of torch that works for you
```

The ManiSkill/SAPIEN simulator code is dependent on working NVIDIA drivers and vulkan packages. After running pip install above, if something is wrong with drivers/vulkan, please follow the troubleshooting guide here: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#troubleshooting

To double check if the simulator is installed correctly, you can run 

```
python -m mani_skill.examples.demo_random_action
```

Then we install lerobot which enable ease of use with all kinds of hardware.

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e .
```

Note that depending on what hardware you are using you might need to install additional packages in LeRobot. If you already installed lerobot somewhere else you can use that instead of running the command above.

## What I have changed in the sim2 real package (You may need to check robot config in real_robot.py)

### 1. In ppo_seg.py, I have modify the CNN input, and the forward function. Few lines to make the network accepts segmentation tensor, similar to previous framework which accept rgb tensor
### 2. demo_vis_seg_only: Script to view your env in cameras segmentation view
### 3. eval_ppo_seg_v2: This contains two thread running simutaneously, the higher frequency video streaming/yoloE loop and the lower frequency robot control loop. For video streaming loop see test_yolo. For robot control loop see eval_ppo_rgb.py which is the original rgb evaluation
### 4. Demo_SO100_pickcube: Script which loads the grasp_cube env and allows you to keyboard teleop the robot.

## What I have changed in the maniskill package
### 1. grasp_cube.py: [train & deploy]
robot pose elevated 7cm to match xlerobot height
Added "segmentation"in SUPPORTED_OBS_MODES

### 2. sim2real_env.py: [deploy]
def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
	# Skip sensor data processing since we're using direct camera access for YOLO streaming
	# Only state data is needed from real environment
	return {}
	
### 3. agent/robots/lerobot/manipulator: [deploy]
def capture_sensor_data(self, sensor_names: Optional[List[str]] = None):
	# Commented out since we're using direct camera access for YOLO streaming
	# Only state data is needed from real environment
	pass

def get_sensor_data(self, sensor_names: Optional[List[str]] = None):
	# Commented out since we're using direct camera access for YOLO streaming
	# Only state data is needed from real environment
	return {}
	
### 4. agent/robots/so100/so_100.py: [train & deploy]
Added a in-hand camera. You can modify the camera setting like fov, pos, etc.

### 5. asset/robots/so100: [train & deploy]
Update robot meshes
Updatr robot urdf

### 6. utils/wrappers/flatten.py: [train & deploy]
Added custom class FlattenRGBDSegmentationObservationWrapper(). This mimics other RGBD wrapper framework and allows user to separate segmentation tensor.


