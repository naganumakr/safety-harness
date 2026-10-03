"""One-time milestone + potential-based-shaping reward for G1's from-scratch (no BC, no warm-start,
no expert-sampled starts beyond a minimal allowance) E2E RL stacking task.

Port of the Franka cube-stack team's own validated recipe (PR #42, KAN-48, commit 95cca31 + the
6fa1617 `placed`-alignment fix) rather than re-deriving it -- see that file's own
`milestone_stack_reward` docstring for the full history: pure milestone-only was undiscoverable
(nothing ever pays until the whole chain is found by luck), and the project's own earlier
continuous/shaped rewards (both here and on Franka) taught "an early lucky grasp followed by an
inevitable bad release scores net negative, so don't grasp at all" -- see rl_rewards_g1.py's own
module docstring for that exact G1-specific lesson. Milestone bonuses are one-time only (nothing to
farm by lingering), and the one continuous term is real Ng/Harada/Russell (ICML 1999) potential-based
shaping, `potential_scale * (gamma*Phi(s') - Phi(s))`: this telescopes to a bounded total along any
trajectory and provably cannot change the optimal policy, unlike the naive "pays every step" terms
that caused the reward-hacking this design is replacing.

KAN-44 lesson folded in directly: `placed` requires real fingertip clearance (see rl_rewards_g1's
LEFT_HAND_LINKS), not wrist-only -- the same bug that made the BC-pipeline's "success" undiscoverable
at the real acceptance bar would just as easily recur here if left on the wrist.

Kernel widths (`reach_kernel`, `over_kernel`) are NOT copied from Franka's 0.4/0.2 -- different robot,
different table geometry. Measure G1's own real cold-start wrist-to-object distance distribution
(diagnose_cold_start_distances.py) before training, the same lesson Franka's own narrow-kernel
saturation failure already paid for once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from isaaclab_tasks.contrib.locomanip_pick_place.mdp.rl_rewards_g1 import (
    LEFT_HAND_JOINT_NAMES,
    LEFT_HAND_LINKS,
    _finger_closed_frac,
    _left_hand_min_dist,
    _left_wrist_pos,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

NAMES = ("reach", "grasp", "lift", "over", "placed")


def _as_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


class milestone_stack_reward_g1(ManagerTermBase):
    """One-time milestone bonuses (reach/grasp/lift/over/placed) plus a per-step completion reward
    while placed holds, plus optional potential-based shaping. See module docstring."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, d = env.num_envs, env.device
        self._steps = torch.zeros(n, device=d)
        self._held_lifted_steps = torch.zeros(n, device=d)
        self._done = torch.zeros(n, len(NAMES), dtype=torch.bool, device=d)
        self._complete_ever = torch.zeros(n, dtype=torch.bool, device=d)
        self._complete_now = torch.zeros(n, dtype=torch.bool, device=d)
        self._phi_prev = torch.zeros(n, device=d)
        self._have_prev = torch.zeros(n, dtype=torch.bool, device=d)

        robot: Articulation = env.scene["robot"]
        joint_ids, _ = robot.find_joints(list(LEFT_HAND_JOINT_NAMES))
        self._joint_ids = joint_ids
        limits = robot.data.soft_joint_pos_limits.torch[0, joint_ids]
        lo, hi = limits[:, 0], limits[:, 1]
        self._open_vals = torch.clamp(torch.zeros_like(lo), lo, hi)
        self._closed_vals = torch.where(lo.abs() > hi.abs(), lo, hi)
        self._hand_link_ids = [robot.data.body_names.index(name) for name in LEFT_HAND_LINKS]

    def reset(self, env_ids: torch.Tensor):
        env_ids = _as_env_ids(self._env, env_ids)
        self._have_prev[env_ids] = False
        if len(env_ids) > 0:
            log = self._env.extras.setdefault("log", {})
            ever = self._done[env_ids].float()
            for i, name in enumerate(NAMES):
                log[f"Milestone/{name}_ever"] = ever[:, i].mean().item()
                if i > 0:
                    log[f"Milestone/{name}_given_prev"] = (
                        ever[:, i].sum() / ever[:, i - 1].sum().clamp(min=1.0)
                    ).item()
            log["Milestone/complete_ever"] = self._complete_ever[env_ids].float().mean().item()
            log["Milestone/complete_at_end"] = self._complete_now[env_ids].float().mean().item()
            log["Milestone/held_lifted_frac_steps"] = (
                self._held_lifted_steps[env_ids] / self._steps[env_ids].clamp(min=1)
            ).mean().item()
            stuck = self._done[env_ids, 3] & ~self._complete_ever[env_ids]
            log["Milestone/reached_over_never_completed"] = stuck.float().mean().item()
        self._steps[env_ids] = 0.0
        self._held_lifted_steps[env_ids] = 0.0
        self._done[env_ids] = False
        self._complete_ever[env_ids] = False
        self._complete_now[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reach_dist: float = 0.20,
        grasp_diff_threshold: float = 0.14,
        finger_closed_threshold: float = 0.5,
        minimal_lift_height: float = 0.03,
        over_xy: float = 0.04,
        hand_clear_dist: float = 0.11,
        reach_kernel: float = 0.3,
        over_kernel: float = 0.1,
        lift_potential_cap: float = 0.06,
        wrist_link_name: str = "left_wrist_yaw_link",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        place_asset_cfg: SceneEntityCfg = SceneEntityCfg("block_b"),
        place_height: float = 0.045,
        milestone_values: tuple[float, float, float, float, float] = (0.1, 0.2, 0.4, 0.5, 0.8),
        completion_reward: float = 2.0,
        potential_scale: float = 0.0,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        object_: RigidObject = env.scene[object_cfg.name]
        base: RigidObject = env.scene[place_asset_cfg.name]
        robot: Articulation = env.scene[robot_cfg.name]
        o = env.scene.env_origins
        object_pos = object_.data.root_pos_w.torch - o
        base_pos = base.data.root_pos_w.torch - o
        wrist_pos = _left_wrist_pos(env, robot_cfg, wrist_link_name)
        target = base_pos.clone()
        target[:, 2] += place_height

        dist_wrist_obj = torch.linalg.norm(object_pos - wrist_pos, dim=1)
        finger_frac = _finger_closed_frac(env, robot_cfg, self._joint_ids, self._open_vals, self._closed_vals)
        reach = dist_wrist_obj < reach_dist
        grasp = (dist_wrist_obj < grasp_diff_threshold) & (finger_frac > finger_closed_threshold)
        rest_z = env.cfg.scene.object.init_state.pos[2]
        lifted = grasp & (object_pos[:, 2] > rest_z + minimal_lift_height)
        dist_xy_target = torch.linalg.norm((object_pos - target)[:, :2], dim=1)
        over = lifted & (dist_xy_target < over_xy)
        hand_clear = _left_hand_min_dist(env, robot_cfg, self._hand_link_ids, object_pos)
        base_rest_z = env.cfg.scene.block_b.init_state.pos[2]
        base_on_table = (base_pos[:, 2] - base_rest_z).abs() < 0.01
        settled = torch.linalg.norm(object_.data.root_vel_w.torch[:, :3], dim=1) < 0.03
        placed = (
            (dist_xy_target < over_xy)
            & (object_pos[:, 2] - target[:, 2]).abs().lt(0.012)
            & base_on_table
            & settled
            & ~grasp
            & (hand_clear > hand_clear_dist)  # KAN-44: fingertip clearance, not wrist-only
        )

        milestones = torch.stack((reach, grasp, lifted, over, placed), dim=1)
        vals = torch.tensor(milestone_values, device=env.device)
        new = milestones & ~self._done
        reward = (new.float() * vals).sum(dim=1)
        self._done |= milestones

        zrel = (object_pos[:, 2] - rest_z).clamp(0.0, lift_potential_cap) / lift_potential_cap
        phi_raw = (
            0.2 * (1.0 - torch.tanh(dist_wrist_obj / reach_kernel))
            + 0.2 * grasp.float()
            + 0.3 * grasp.float() * zrel
            + 0.3 * lifted.float() * (1.0 - torch.tanh(dist_xy_target / over_kernel))
        )
        phi = torch.where(placed, torch.ones_like(phi_raw), phi_raw)

        if potential_scale > 0.0:
            shaping = gamma * phi - self._phi_prev
            reward = reward + potential_scale * torch.where(self._have_prev, shaping, torch.zeros_like(shaping))
        self._phi_prev = phi
        self._have_prev[:] = True

        reward = reward + completion_reward * placed.float()
        self._steps += 1.0
        self._held_lifted_steps += lifted.float()
        self._complete_now = placed
        self._complete_ever |= placed
        return reward
