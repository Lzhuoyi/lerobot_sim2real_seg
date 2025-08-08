"""Simple script to train a RGB PPO policy in simulation

Note: This script has been optimized for GPU memory usage by:
1. Using minimal shader pack to reduce GPU memory
2. Reducing camera resolution to 128x128
3. Reducing default num_envs from 512 to 256

If you still encounter GPU memory issues, try:
- Further reducing num_envs (e.g., 128 or 64)
- Reducing camera resolution further
- Using CPU backend instead of GPU

# Example command:
seed=3
python lerobot_sim2real/scripts/train_ppo_rgb.py --env-id="SO100GraspCube-v1" --env-kwargs-json-path=env_config.json \
  --ppo.seed=${seed} \
  --ppo.num_envs=64 --ppo.num-steps=16 --ppo.update_epochs=8 --ppo.num_minibatches=32 \
  --ppo.total_timesteps=100_000_000 --ppo.gamma=0.9 \
  --ppo.num_eval_envs=8 --ppo.num-eval-steps=64 --ppo.no-partial-reset \
  --ppo.exp-name="ppo-SO100GraspCube-v1-rgb-${seed}" \
  --ppo.track --ppo.wandb_project_name "SO100-ManiSkill"
"""

from dataclasses import dataclass, field
import json
from typing import Optional
import tyro

from lerobot_sim2real.rl.ppo_rgb import PPOArgs, train


@dataclass
class Args:
    env_id: str
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    ppo: PPOArgs = field(default_factory=PPOArgs)
    """PPO training arguments"""

def main(args: Args):
    args.ppo.env_id = args.env_id
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs = json.load(f)
        args.ppo.env_kwargs = env_kwargs
    else:
        print("No env kwargs json path provided, using default env kwargs with default settings")
    train(args=args.ppo)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)