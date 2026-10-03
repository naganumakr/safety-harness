"""Measure G1's real cold-start wrist-to-object and object-to-target distance distributions, from a
genuinely random reset (no policy, no snapshots) -- before picking reach_kernel/over_kernel widths
for rl_rewards_g1_sparse.milestone_stack_reward_g1.

Franka's own first potential-shaping attempt gave no usable gradient (reach/grasp/lift stayed near
0%) because its tanh kernel saturated past the actual cold-start distance range; the fix was widening
the kernel to the real scale, not changing the formula. Measuring first instead of reusing Franka's
0.4/0.2 (different robot, different table geometry) avoids rediscovering that same bug independently.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Stack-Blocks-G1-RL-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.contrib.locomanip_pick_place.g1_block_stack_rl_env_cfg import BLOCK  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _pct(t: torch.Tensor, q: float) -> float:
    return float(torch.quantile(t, q))


env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
# XR-teleop leftover, unconditionally defined on the base scene cfg; crashes sensor init unless
# enable_cameras is set (same bug eval_policy.py already works around).
env_cfg.scene.robot_pov_cam = None
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

with torch.inference_mode():
    env.reset()
    o = env.scene.env_origins
    robot = env.scene["robot"]
    object_ = env.scene["object"]
    block_b = env.scene["block_b"]
    widx = robot.data.body_names.index("left_wrist_yaw_link")

    wrist_pos = robot.data.body_pos_w.torch[:, widx] - o
    object_pos = object_.data.root_pos_w.torch - o
    base_pos = block_b.data.root_pos_w.torch - o
    target = base_pos.clone()
    target[:, 2] += BLOCK

    d_wrist_obj = torch.linalg.norm(object_pos - wrist_pos, dim=1)
    d_obj_target_xy = torch.linalg.norm((object_pos - target)[:, :2], dim=1)

    print(f"N={env.num_envs} fresh resets (no policy, no snapshots), task={args_cli.task}")
    print("wrist-to-object distance at reset (m):")
    for q in (0.0, 0.1, 0.5, 0.9, 1.0):
        print(f"  p{int(q*100):>3}: {_pct(d_wrist_obj, q):.4f}")
    print(f"  mean={float(d_wrist_obj.mean()):.4f} std={float(d_wrist_obj.std()):.4f}")
    print("object-to-target xy distance at reset (m) -- only meaningful once lifted, reset-time value is a baseline:")
    for q in (0.0, 0.1, 0.5, 0.9, 1.0):
        print(f"  p{int(q*100):>3}: {_pct(d_obj_target_xy, q):.4f}")
    print(f"  mean={float(d_obj_target_xy.mean()):.4f} std={float(d_obj_target_xy.std()):.4f}")
    print(
        "\nSuggested kernel widths (so 1-tanh(d/kernel) still carries real gradient near the p90 cold-start "
        "distance, not just near 0): reach_kernel ~ p90(wrist-to-object)/1.5, over_kernel from a lifted-carry "
        "sweep instead (this script's object-to-target number is reset-time only, not post-lift)."
    )

simulation_app.close()
