"""Computed-torque action term for RoboCrane.

Policy outputs normalized joint accelerations in [-1, 1].
These are denormalized, integrated at physics rate, and converted to torques
with a computed-torque style law:
  tau = M_hat * (qddot_ff + Kp (q_d - q) + Kd (qdot_d - qdot)) + bias
where M_hat is a configurable diagonal approximation and bias is taken from
MuJoCo's bias forces (if available).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


@dataclass(kw_only=True)
class JointAccelerationCtcActionCfg(ActionTermCfg):
  joint_names: tuple[str, ...]
  qddot_limits: tuple[float, ...]
  qdot_limits: tuple[float, ...]
  torque_limits: tuple[float, ...]
  kp: tuple[float, ...]
  kd: tuple[float, ...]
  inertia_diag: tuple[float, ...]

  def build(self, env) -> "JointAccelerationCtcAction":
    return JointAccelerationCtcAction(self, env)


class JointAccelerationCtcAction(ActionTerm):
  cfg: JointAccelerationCtcActionCfg

  def __init__(self, cfg: JointAccelerationCtcActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    joint_ids, joint_names = self._entity.find_joints(cfg.joint_names, preserve_order=True)
    if not joint_ids:
      raise ValueError(
        f"No joints matched {cfg.joint_names}. Available joints: {self._entity.joint_names}"
      )

    self._joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
    self._joint_names = joint_names
    self._action_dim = len(joint_ids)

    def _vec(values: tuple[float, ...], name: str) -> torch.Tensor:
      if len(values) != self._action_dim:
        raise ValueError(
          f"{name} length must be {self._action_dim}, got {len(values)}."
        )
      return torch.tensor(values, dtype=torch.float32, device=self.device).view(1, -1)

    self._qddot_lim = _vec(cfg.qddot_limits, "qddot_limits")
    self._qdot_lim = _vec(cfg.qdot_limits, "qdot_limits")
    self._tau_lim = _vec(cfg.torque_limits, "torque_limits")
    self._kp = _vec(cfg.kp, "kp")
    self._kd = _vec(cfg.kd, "kd")
    self._mhat = _vec(cfg.inertia_diag, "inertia_diag")

    self._raw_action = torch.zeros(
      (self.num_envs, self._action_dim), dtype=torch.float32, device=self.device
    )
    self._qddot_cmd = torch.zeros_like(self._raw_action)
    self._qdot_d = torch.zeros_like(self._raw_action)
    self._q_d = torch.zeros_like(self._raw_action)

    joint_pos = self._entity.data.joint_pos[:, self._joint_ids]
    self._q_d.copy_(joint_pos)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_action[:] = torch.clamp(actions, -1.0, 1.0)
    self._qddot_cmd[:] = self._raw_action * self._qddot_lim

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_action[env_ids] = 0.0
    self._qddot_cmd[env_ids] = 0.0
    self._qdot_d[env_ids] = 0.0
    joint_pos = self._entity.data.joint_pos[:, self._joint_ids]
    self._q_d[env_ids] = joint_pos[env_ids]

  def apply_actions(self) -> None:
    dt = self._env.physics_dt

    q = self._entity.data.joint_pos[:, self._joint_ids]
    qdot = self._entity.data.joint_vel[:, self._joint_ids]

    # Substep integration of acceleration command -> desired velocity/position.
    self._qdot_d += dt * self._qddot_cmd
    self._qdot_d = torch.clamp(self._qdot_d, -self._qdot_lim, self._qdot_lim)

    self._q_d += dt * self._qdot_d
    q_limits = self._entity.data.joint_pos_limits[:, self._joint_ids]
    self._q_d = torch.clamp(self._q_d, q_limits[..., 0], q_limits[..., 1])

    q_err = self._q_d - q
    qdot_err = self._qdot_d - qdot
    qddot_ref = self._qddot_cmd + self._kp * q_err + self._kd * qdot_err

    # Bias compensation from MuJoCo dynamics (Coriolis + gravity + etc.).
    bias = torch.zeros_like(qddot_ref)
    raw_data = self._entity.data.data
    if hasattr(raw_data, "qfrc_bias"):
      v_adr = self._entity.indexing.joint_v_adr[self._joint_ids]
      bias = raw_data.qfrc_bias[:, v_adr]

    tau = self._mhat * qddot_ref + bias
    tau = torch.clamp(tau, -self._tau_lim, self._tau_lim)

    self._entity.set_joint_effort_target(tau, joint_ids=self._joint_ids)
