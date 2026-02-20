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

import mujoco_warp as mjwarp
import torch
import warp as wp

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
  action_lpf_alpha: float = 0.85
  ee_site_name: str = "gripping_point"
  wrench_damping: float = 1.0e-4
  torque_lpf_alpha: float = 0.9

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
    self._joint_dof_ids = self._entity.indexing.joint_v_adr[self._joint_ids]
    self._nv = int(self._env.sim.mj_model.nv)

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
    if not 0.0 <= cfg.action_lpf_alpha < 1.0:
      raise ValueError("action_lpf_alpha must be in [0, 1).")
    self._action_lpf_alpha = float(cfg.action_lpf_alpha)
    if not 0.0 <= cfg.torque_lpf_alpha < 1.0:
      raise ValueError("torque_lpf_alpha must be in [0, 1).")
    self._torque_lpf_alpha = float(cfg.torque_lpf_alpha)
    self._wrench_damping = float(cfg.wrench_damping)
    if self._wrench_damping <= 0.0:
      raise ValueError("wrench_damping must be > 0.")

    site_ids, _ = self._entity.find_sites(cfg.ee_site_name, preserve_order=True)
    if not site_ids:
      raise ValueError(
        f"No site matched '{cfg.ee_site_name}'. Available sites: {self._entity.site_names}"
      )
    self._ee_site_local_id = int(site_ids[0])
    self._ee_site_id = int(self._entity.indexing.site_ids[self._ee_site_local_id].item())
    self._ee_body_id = int(self._env.sim.mj_model.site_bodyid[self._ee_site_id])

    self._raw_action = torch.zeros(
      (self.num_envs, self._action_dim), dtype=torch.float32, device=self.device
    )
    self._raw_action_filtered = torch.zeros_like(self._raw_action)
    self._qddot_cmd = torch.zeros_like(self._raw_action)
    self._qdot_d = torch.zeros_like(self._raw_action)
    self._q_d = torch.zeros_like(self._raw_action)
    self._tau_cmd = torch.zeros_like(self._raw_action)
    self._tau_cmd_filtered = torch.zeros_like(self._raw_action)
    self._tau_nominal = torch.zeros_like(self._raw_action)
    self._tau_residual = torch.zeros_like(self._raw_action)
    self._wrench_ext = torch.zeros(
      (self.num_envs, 6), dtype=torch.float32, device=self.device
    )
    self._force_ext = self._wrench_ext[:, :3]
    self._force_norm = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
    self._tau_res_norm = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
    self._qacc_backup = torch.zeros(
      (self.num_envs, self._nv), dtype=torch.float32, device=self.device
    )

    joint_pos = self._entity.data.joint_pos[:, self._joint_ids]
    self._q_d.copy_(joint_pos)

    with wp.ScopedDevice(self._env.sim.wp_device):
      self._jacp_wp = wp.zeros((self.num_envs, 3, self._nv), dtype=float)
      self._jacr_wp = wp.zeros((self.num_envs, 3, self._nv), dtype=float)
      self._point_wp = wp.zeros(self.num_envs, dtype=wp.vec3)
      self._body_wp = wp.zeros(self.num_envs, dtype=wp.int32)
      self._body_wp.fill_(self._ee_body_id)
    self._jacp_torch = wp.to_torch(self._jacp_wp)
    self._jacr_torch = wp.to_torch(self._jacr_wp)
    self._point_torch = wp.to_torch(self._point_wp).view(self.num_envs, 3)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  @property
  def tau_nominal(self) -> torch.Tensor:
    return self._tau_nominal

  @property
  def tau_residual(self) -> torch.Tensor:
    return self._tau_residual

  @property
  def force_ext(self) -> torch.Tensor:
    return self._force_ext

  @property
  def wrench_ext(self) -> torch.Tensor:
    return self._wrench_ext

  @property
  def force_norm(self) -> torch.Tensor:
    return self._force_norm

  @property
  def tau_res_norm(self) -> torch.Tensor:
    return self._tau_res_norm

  @property
  def q_d(self) -> torch.Tensor:
    return self._q_d

  @property
  def qdot_d(self) -> torch.Tensor:
    return self._qdot_d

  @property
  def qddot_d(self) -> torch.Tensor:
    return self._qddot_cmd

  @property
  def tau_cmd(self) -> torch.Tensor:
    return self._tau_cmd

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_action[:] = torch.clamp(actions, -1.0, 1.0)
    a = self._action_lpf_alpha
    self._raw_action_filtered[:] = (
      a * self._raw_action_filtered + (1.0 - a) * self._raw_action
    )
    self._qddot_cmd[:] = self._raw_action_filtered * self._qddot_lim

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_action[env_ids] = 0.0
    self._raw_action_filtered[env_ids] = 0.0
    self._qddot_cmd[env_ids] = 0.0
    self._qdot_d[env_ids] = 0.0
    self._tau_cmd[env_ids] = 0.0
    self._tau_cmd_filtered[env_ids] = 0.0
    self._tau_nominal[env_ids] = 0.0
    self._tau_residual[env_ids] = 0.0
    self._wrench_ext[env_ids] = 0.0
    self._force_norm[env_ids] = 0.0
    self._tau_res_norm[env_ids] = 0.0
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
    self._tau_cmd[:] = tau
    a_tau = self._torque_lpf_alpha
    self._tau_cmd_filtered[:] = a_tau * self._tau_cmd_filtered + (1.0 - a_tau) * tau

    self._tau_nominal[:] = self._compute_nominal_torque(qddot_ref)
    self._tau_residual[:] = self._tau_cmd_filtered - self._tau_nominal
    self._tau_res_norm[:] = torch.linalg.norm(self._tau_residual, dim=-1)
    self._wrench_ext[:] = self._estimate_external_wrench(self._tau_residual)
    self._force_norm[:] = torch.linalg.norm(self._force_ext, dim=-1)

    self._entity.set_joint_effort_target(tau, joint_ids=self._joint_ids)

  def _compute_nominal_torque(self, qddot_des: torch.Tensor) -> torch.Tensor:
    raw_data = self._entity.data.data
    self._qacc_backup[:] = raw_data.qacc
    raw_data.qacc[:] = 0.0
    raw_data.qacc[:, self._joint_dof_ids] = qddot_des
    with wp.ScopedDevice(self._env.sim.wp_device):
      mjwarp.inverse(self._env.sim.wp_model, self._env.sim.wp_data)
    tau_model = raw_data.qfrc_inverse[:, self._joint_dof_ids]
    raw_data.qacc[:] = self._qacc_backup
    return tau_model

  def _estimate_external_wrench(self, tau_res: torch.Tensor) -> torch.Tensor:
    self._point_torch[:] = self._entity.data.site_pos_w[:, self._ee_site_local_id]
    with wp.ScopedDevice(self._env.sim.wp_device):
      mjwarp.jac(
        self._env.sim.wp_model,
        self._env.sim.wp_data,
        self._jacp_wp,
        self._jacr_wp,
        self._point_wp,
        self._body_wp,
      )

    jacp = self._jacp_torch[:, :, self._joint_dof_ids]
    jacr = self._jacr_torch[:, :, self._joint_dof_ids]
    jac6 = torch.cat([jacp, jacr], dim=1)  # (n_env, 6, n_joints)
    jt = jac6.transpose(1, 2)  # (n_env, n_joints, 6)
    eye = torch.eye(self._action_dim, device=self.device, dtype=jac6.dtype).unsqueeze(0)
    a = jt @ jt.transpose(1, 2) + self._wrench_damping * eye
    j_pinv_t = torch.linalg.solve(a, jt)
    wrench = j_pinv_t.transpose(1, 2) @ tau_res.unsqueeze(-1)
    return wrench.squeeze(-1)
