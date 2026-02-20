"""RoboCrane asset configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

_ROOT = Path(__file__).resolve().parents[1]
ROBOCRANE_XML = _ROOT / "rl_robocrane" / "robocrane" / "robocrane_mjlab_clean.xml"


def get_assets(meshdir: str | None) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ROBOCRANE_XML.parent, meshdir=meshdir, recursive=False)
  return assets


def get_spec() -> mujoco.MjSpec:
  # Some scene camera attributes in this XML are accepted by mjModel loaders but not
  # by MjSpec parser. Strip them for MjSpec-based task construction.
  xml_text = ROBOCRANE_XML.read_text()
  xml_text = xml_text.replace(' orthographic="true"', "")
  xml_text = xml_text.replace(' orthographic="false"', "")
  spec = mujoco.MjSpec.from_string(xml_text)
  # Normalize names by dropping namespace prefixes so mjlab actuator/action name
  # resolution remains consistent with robots that use unqualified names.
  for joint in spec.joints:
    if joint.name:
      joint.name = joint.name.split("/")[-1]
  for actuator in spec.actuators:
    if actuator.target:
      actuator.target = actuator.target.split("/")[-1]
  spec.assets = get_assets(spec.meshdir)
  return spec


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint_1": 0.0,
    "joint_2": 0.0,
    "joint_3": 0.0,
    "joint_4": -1.5708,
    "joint_5": 0.0,
    "joint_6": 1.5708,
    "joint_7": 0.0,
    "joint_cj1": 0.0,
    "joint_cj2": 0.0,
  },
  joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlMotorActuatorCfg(target_names_expr=("joint_[1-7]",)),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_robocrane_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )
