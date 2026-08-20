#!/usr/bin/env python3
"""
SO100 (SO-ARM100) MuJoCo teleoperation with a Nintendo Joy-Con.

Loads the standalone SO100 arm (``so100_scene.xml``) in MuJoCo and drives its
end effector in Cartesian space from a Joy-Con, using the same analytic 2-link
IK as the real-robot examples in ``software/examples/``.

The Joy-Con mapping follows leisaac's ``SO101JoyConEE`` device
(``leisaac/devices/gamepad/joycon_ee_gamepad.py``), the ``joycon-ee`` option of
``teleop_se3_agent2.py``: the stick drives the end effector along the direction
the gripper faces, the lateral offset drives ``shoulder_pan`` linearly, and the
tilt sets the gripper pitch and wrist roll. leisaac steers by the Joy-Con's own
attitude here; this version follows the arm, so forward always means "further
along the way the jaws point".

Usage
-----
    python so100_joycon_mujoco.py                 # Joy-Con, fall back to keyboard
    python so100_joycon_mujoco.py --device joycon # require a Joy-Con
    python so100_joycon_mujoco.py --device keyboard
    python so100_joycon_mujoco.py --seed 0        # reproducible block placement
    python so100_joycon_mujoco.py --fixed-block   # block stays where the scene puts it
    python so100_joycon_mujoco.py --selftest      # headless IK/FK check, no viewer

The block spawns at a random spot inside BLOCK_FWD_RANGE / BLOCK_LAT_RANGE with a
random yaw; edit those constants to change the sampling box.

Joy-Con (right) controls
------------------------
    stick up/down     : end effector along the gripper's facing direction (x+z)
    stick left/right  : end effector left / right (drives shoulder_pan)
    R                 : end effector up
    stick press       : end effector down
    X / B             : fine forward / backward
    tilt the Joy-Con  : gripper pitch (nose up/down) and wrist roll
    ZR                : toggle gripper open/close
    HOME              : reset the end effector position
    PLUS              : re-calibrate the Joy-Con IMU and reset the position

Keyboard controls (viewer window must have focus)
-------------------------------------------------
    W/S forward/back   A/D left/right   R/F up/down
    T/G pitch up/down  Y/H roll +/-     SPACE toggle gripper   0 reset
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_SCENE = os.path.join(HERE, "so100_scene.xml")

# ``joyconrobotics`` is vendored in the repo, not installed as a package.
if os.path.join(REPO_ROOT, "software") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "software"))

# ----------------------------------------------------------------------------
# Robot geometry (metres / radians), taken from so100.xml
# ----------------------------------------------------------------------------
# The shoulder-pan axis is the vertical line through the Rotation_Pitch body.
PAN_AXIS_XY = (0.0, -0.0452)
# Offset of the shoulder-pitch axis from the pan axis, expressed in the rotated
# arm frame: +forward and +up.
SHOULDER_FWD = 0.0306
SHOULDER_UP = 0.1190
# At shoulder_pan = 0 the arm points along world -Y; "left" is world +X.
FORWARD_WORLD = np.array([0.0, -1.0, 0.0])
LEFT_WORLD = np.array([1.0, 0.0, 0.0])

L1 = 0.1159  # upper arm
L2 = 0.1350  # lower arm

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_LIMITS = {
    "shoulder_pan": (-2.1, 2.1),
    "shoulder_lift": (-0.1, 3.45),
    "elbow_flex": (-0.2, 3.14159),
    "wrist_flex": (-2.8, 2.8),
    "wrist_roll": (-3.14159, 3.14159),
    "gripper": (-0.2, 2.0),
}

# ----------------------------------------------------------------------------
# Joy-Con -> joint mapping, ported from leisaac's ``SO101JoyConEE``
# ----------------------------------------------------------------------------
# leisaac emits motor-space degrees, which ``convert_action_from_so101_leader``
# rescales from SO101_FOLLOWER_MOTOR_LIMITS into SO101_FOLLOWER_USD_JOINT_LIMLITS
# before converting to radians. The constants below fold both steps together, so
# the same Joy-Con motion produces the same joint angle as it does in Isaac.
#
#   shoulder_pan  = 300 deg/m  * lateral, motor (-100,100) -> usd (-110,110)
#   wrist_roll    =  50 deg/rad * roll,   motor (-100,100) -> usd (-160,160)
#   gripper pitch = 300 deg/rad * pitch - 20 deg bias, usd scale 190/200
#   gripper       = 60 / 20 motor deg,    motor (0,100)    -> usd (-10,100)
PAN_PER_LAT = math.radians(300.0 * 1.1)  # rad of shoulder_pan per metre of lateral offset
PITCH_GAIN = math.radians(300.0 * 0.95)  # rad of gripper pitch per rad of Joy-Con tilt
PITCH_OFFSET = math.radians(20.0 * 0.95)  # nose-down bias with the Joy-Con held level
ROLL_GAIN = math.radians(50.0 * 1.6)  # rad of wrist_roll per rad of Joy-Con roll

GRIPPER_OPEN = math.radians(60.0 * 1.1 - 10.0)
GRIPPER_CLOSED = math.radians(20.0 * 1.1 - 10.0)

# Home end-effector pose, relative to the pan axis (forward, lateral, height).
# The offsets match leisaac's rest-pose calibration (``_x0`` / ``_z0``).
HOME_FWD = SHOULDER_FWD + 0.1629
HOME_LAT = 0.0
HOME_UP = SHOULDER_UP + 0.1131
HOME_PITCH = -PITCH_OFFSET  # gripper pitch, rad, 0 = horizontal, + = nose up
HOME_ROLL = 0.0

# Cartesian workspace clamp (metres), keeps the IK inside a sane region.
FWD_RANGE = (0.06, 0.34)
LAT_RANGE = (-0.28, 0.28)
UP_RANGE = (0.02, 0.42)
PITCH_RANGE = (-2.8, 2.8)  # matches the wrist_flex limit, so PITCH_GAIN is not clipped

# ----------------------------------------------------------------------------
# Random block placement
# ----------------------------------------------------------------------------
# Where the pick-and-place block spawns, in the same arm-centred frame as the
# teleop command: forward from the shoulder-pan axis, and lateral (+ = left).
# The arm reaches SHOULDER_FWD + L1 + L2 = 0.281 m at shoulder height and less
# near the floor, so keep the forward range comfortably inside that.
BLOCK_BODY = "block"
BLOCK_JOINT = "block_free"
BLOCK_FWD_RANGE = (0.13, 0.23)
BLOCK_LAT_RANGE = (-0.11, 0.11)
BLOCK_YAW_RANGE = (-math.pi, math.pi)
BLOCK_Z = 0.030  # half the block height, so it rests on the floor


def clamp(value, low, high):
    return max(low, min(high, value))


# ----------------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------------
def inverse_kinematics_rad(x, y, l1=L1, l2=L2):
    """Planar 2-link IK for the SO100 shoulder/elbow pair.

    ``x`` is the horizontal reach and ``y`` the height of the wrist centre,
    both measured from the shoulder-pitch axis. Returns
    ``(shoulder_lift, elbow_flex)`` in radians, directly usable as MuJoCo
    joint targets.
    """
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    r = math.hypot(x, y)
    r_max = l1 + l2
    if r > r_max:
        scale = r_max / r
        x, y, r = x * scale, y * scale, r_max

    r_min = abs(l1 - l2)
    if 0 < r < r_min:
        scale = r_min / r
        x, y, r = x * scale, y * scale, r_min

    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = math.pi - math.acos(clamp(cos_theta2, -1.0, 1.0))

    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    shoulder_lift = clamp(theta1 + theta1_offset, *JOINT_LIMITS["shoulder_lift"])
    elbow_flex = clamp(theta2 + theta2_offset, *JOINT_LIMITS["elbow_flex"])
    return shoulder_lift, elbow_flex


def ee_to_joints(fwd, lat, up, pitch, roll):
    """Map a Cartesian end-effector command to the six SO100 joint targets.

    ``fwd``/``lat``/``up`` are metres relative to the shoulder-pan axis;
    ``pitch`` is the gripper pitch in radians (0 = horizontal, + = nose up).

    Like leisaac, the lateral offset drives ``shoulder_pan`` linearly instead of
    pointing the arm at a Cartesian target, so the IK sees ``fwd`` alone as the
    reach. The sign is negated relative to leisaac's ``shoulder_pan = y * 300``
    so that pushing the stick left still moves the gripper left in this scene.
    """
    pan = clamp(-PAN_PER_LAT * lat, *JOINT_LIMITS["shoulder_pan"])

    reach = fwd - SHOULDER_FWD
    height = up - SHOULDER_UP
    shoulder_lift, elbow_flex = inverse_kinematics_rad(reach, height)

    # In this model the gripper pitch is shoulder_lift - elbow_flex - wrist_flex.
    wrist_flex = clamp(shoulder_lift - elbow_flex - pitch, *JOINT_LIMITS["wrist_flex"])
    wrist_roll = clamp(roll, *JOINT_LIMITS["wrist_roll"])
    return {
        "shoulder_pan": pan,
        "shoulder_lift": shoulder_lift,
        "elbow_flex": elbow_flex,
        "wrist_flex": wrist_flex,
        "wrist_roll": wrist_roll,
    }


# ----------------------------------------------------------------------------
# Teleop sources
# ----------------------------------------------------------------------------
class EETarget:
    """Mutable Cartesian end-effector target shared by every teleop source."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.fwd = HOME_FWD
        self.lat = HOME_LAT
        self.up = HOME_UP
        self.pitch = HOME_PITCH
        self.roll = HOME_ROLL
        self.gripper_open = True

    def clamp(self):
        self.fwd = clamp(self.fwd, *FWD_RANGE)
        self.lat = clamp(self.lat, *LAT_RANGE)
        self.up = clamp(self.up, *UP_RANGE)
        self.pitch = clamp(self.pitch, *PITCH_RANGE)

    def joint_targets(self):
        targets = ee_to_joints(self.fwd, self.lat, self.up, self.pitch, self.roll)
        targets["gripper"] = GRIPPER_OPEN if self.gripper_open else GRIPPER_CLOSED
        return targets


def _normalise_joycon_serials():
    """Make Windows Joy-Con serials acceptable to joyconrobotics.

    hidapi reports the serial as a bare MAC (``'862536005a10'``), but
    ``JoyconRobotics.__init__`` only accepts a ``9c:54:`` prefix or a
    colon-separated 17-character MAC and raises ``IOError("There is no joycon
    for robotics")`` otherwise. The serial is never used to open the device, so
    re-formatting it before the check is safe.
    """
    from joyconrobotics import joyconrobotics as jr

    def with_colons(getter):
        @functools.wraps(getter)
        def wrapper(*args, **kwargs):
            vendor_id, product_id, serial = getter(*args, **kwargs)
            if serial and len(serial) == 12 and ":" not in serial:
                serial = ":".join(serial[i : i + 2] for i in range(0, 12, 2))
            return vendor_id, product_id, serial

        wrapper._serial_normalised = True
        return wrapper

    if not getattr(jr.get_R_id, "_serial_normalised", False):
        jr.get_R_id = with_colons(jr.get_R_id)
        jr.get_L_id = with_colons(jr.get_L_id)


class JoyconSource:
    """Joy-Con teleop using leisaac's ``SO101JoyConEE`` mapping."""

    def __init__(self, side="right"):
        _normalise_joycon_serials()
        from joyconrobotics import JoyconRobotics

        class FixedAxesJoycon(JoyconRobotics):
            """Stick mapping of leisaac's ``FixedAxesJoyconRobotics``.

            The vertical stick moves the end effector along the direction the
            gripper currently faces, so it drives x and z together; the
            horizontal stick drives y on its own.
            """

            # Gripper pitch of the last command, written back by
            # ``JoyconSource.apply()``. leisaac steers by ``direction_vector``,
            # the way the *Joy-Con* points; this follows the arm instead.
            gripper_pitch = HOME_PITCH

            def common_update(self):
                speed = 0.0008
                is_right = self.joycon.is_right()

                # unit vector of the gripper in the arm's (forward, up) plane
                pointing_x = math.cos(self.gripper_pitch)
                pointing_z = math.sin(self.gripper_pitch)

                stick_v = (
                    self.joycon.get_stick_right_vertical()
                    if is_right
                    else self.joycon.get_stick_left_vertical()
                )
                if abs(stick_v - 1800) > 300:
                    delta_v = speed * (stick_v - 1800) / 1000
                    self.position[0] += (
                        delta_v * self.dof_speed[0] * self.direction_reverse[0] * pointing_x
                    )
                    # leisaac indexes dof_speed/direction_reverse with 1 here
                    self.position[2] += (
                        delta_v * self.dof_speed[1] * self.direction_reverse[1] * pointing_z
                    )

                stick_h = (
                    self.joycon.get_stick_right_horizontal()
                    if is_right
                    else self.joycon.get_stick_left_horizontal()
                )
                if abs(stick_h - 2000) > 300:
                    self.position[1] += (
                        speed * (stick_h - 2000) / 1000 * self.dof_speed[1]
                        * self.direction_reverse[1]
                    )

                up = self.joycon.get_button_r() if is_right else self.joycon.get_button_l()
                if up == 1:
                    self.position[2] += speed * self.dof_speed[2] * self.direction_reverse[2]
                down = (
                    self.joycon.get_button_r_stick()
                    if is_right
                    else self.joycon.get_button_l_stick()
                )
                if down == 1:
                    self.position[2] -= speed * self.dof_speed[2] * self.direction_reverse[2]

                fine_fwd = self.joycon.get_button_x() if is_right else self.joycon.get_button_up()
                fine_back = self.joycon.get_button_b() if is_right else self.joycon.get_button_down()
                if fine_fwd == 1:
                    self.position[0] += 0.001 * self.dof_speed[0]
                elif fine_back == 1:
                    self.position[0] -= 0.001 * self.dof_speed[0]

                home = (
                    self.joycon.get_button_home() if is_right else self.joycon.get_button_capture()
                )
                if home == 1:
                    self.position = self.offset_position_m.copy()

                for event_type, status in self.button.events():
                    zr_pressed = (is_right and event_type == "zr") or (
                        not is_right and event_type == "zl"
                    )
                    if zr_pressed and status == 1:
                        self.gripper_state = (
                            self.gripper_close
                            if self.gripper_state == self.gripper_open
                            else self.gripper_open
                        )
                    # leisaac maps PLUS to an IMU re-calibration; A / Y drive the
                    # episode recorder there, which has no counterpart here.
                    recalibrate = (is_right and event_type == "plus") or (
                        not is_right and event_type == "minus"
                    )
                    if recalibrate and status == 1:
                        self.position = self.offset_position_m.copy()
                        self.reset_joycon()

                return self.position, self.gripper_state, self.button_control

        self._joycon = FixedAxesJoycon(side, dof_speed=[2, 2, 2, 1, 1, 1])
        self.name = f"joycon-{side}"

    def apply(self, target: EETarget):
        pose, gripper, _ = self._joycon.get_control()
        dx, dy, dz, roll, pitch, _yaw = pose

        target.fwd = HOME_FWD + dx
        target.lat = HOME_LAT + dy
        target.up = HOME_UP + dz
        # leisaac: pitch_deg = -pitch * 300 + 20, folded into the wrist through
        # wrist_flex = shoulder_lift - elbow_flex - pitch.
        target.pitch = pitch * PITCH_GAIN - PITCH_OFFSET
        target.roll = clamp(roll * ROLL_GAIN, *JOINT_LIMITS["wrist_roll"])
        target.gripper_open = gripper == self._joycon.gripper_open
        target.clamp()
        # feed the clamped gripper pitch back so the next stick step travels
        # along the direction the gripper faces
        self._joycon.gripper_pitch = target.pitch

    def close(self):
        try:
            self._joycon.disconnect()
        except Exception:
            pass


class KeyboardSource:
    """Fallback teleop driven by the MuJoCo viewer's key callback."""

    STEP = 0.005
    ANG_STEP = 0.05

    def __init__(self):
        self.name = "keyboard"
        self._pending = []

    def key_callback(self, keycode):
        self._pending.append(keycode)

    def apply(self, target: EETarget):
        pending, self._pending = self._pending, []
        for keycode in pending:
            key = chr(keycode).upper() if 0 < keycode < 0x110000 else ""
            if key == "W":
                target.fwd += self.STEP
            elif key == "S":
                target.fwd -= self.STEP
            elif key == "A":
                target.lat += self.STEP
            elif key == "D":
                target.lat -= self.STEP
            elif key == "R":
                target.up += self.STEP
            elif key == "F":
                target.up -= self.STEP
            elif key == "T":
                target.pitch += self.ANG_STEP
            elif key == "G":
                target.pitch -= self.ANG_STEP
            elif key == "Y":
                target.roll += self.ANG_STEP
            elif key == "H":
                target.roll -= self.ANG_STEP
            elif key == " ":
                target.gripper_open = not target.gripper_open
            elif key == "0":
                target.reset()
        target.clamp()

    def close(self):
        pass


def make_source(device, side):
    """Return a teleop source, falling back to the keyboard when asked to."""
    if device in ("joycon", "auto"):
        try:
            return JoyconSource(side)
        except Exception as exc:  # no Joy-Con paired / hidapi missing / ...
            if device == "joycon":
                raise RuntimeError(
                    f"could not open the {side} Joy-Con ({type(exc).__name__}: {exc}). "
                    "Pair it over Bluetooth first, or run with --device keyboard."
                ) from exc
            print(f"[warn] Joy-Con unavailable ({exc}); falling back to keyboard control.")
    return KeyboardSource()


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
class SO100Sim:
    def __init__(self, scene_path=DEFAULT_SCENE, rng=None):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.act_id = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINTS
        }
        missing = [n for n, i in self.act_id.items() if i < 0]
        if missing:
            raise RuntimeError(f"actuators missing from {scene_path}: {missing}")
        self.qpos_adr = {}
        for name in JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.qpos_adr[name] = self.model.jnt_qposadr[jid]
        self.wrist_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Wrist_Pitch_Roll"
        )
        # the block is optional: a scene without one still teleoperates fine
        block_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, BLOCK_JOINT)
        if block_jid >= 0 and self.model.jnt_type[block_jid] == mujoco.mjtJoint.mjJNT_FREE:
            self.block_qpos_adr = self.model.jnt_qposadr[block_jid]
            self.block_dof_adr = self.model.jnt_dofadr[block_jid]
        else:
            self.block_qpos_adr = None
            self.block_dof_adr = None
        self.reset()

    def reset(self, randomise_block=False):
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        placement = self.place_block_randomly() if randomise_block else None
        mujoco.mj_forward(self.model, self.data)
        return placement

    def place_block_randomly(self):
        """Drop the block at a random spot inside the BLOCK_*_RANGE box.

        Returns the sampled ``(forward, lateral, yaw)``, or ``None`` when the
        scene has no free-floating block.
        """
        if self.block_qpos_adr is None:
            return None

        fwd = float(self.rng.uniform(*BLOCK_FWD_RANGE))
        lat = float(self.rng.uniform(*BLOCK_LAT_RANGE))
        yaw = float(self.rng.uniform(*BLOCK_YAW_RANGE))

        base = np.array([PAN_AXIS_XY[0], PAN_AXIS_XY[1], 0.0])
        pos = base + FORWARD_WORLD * fwd + LEFT_WORLD * lat + np.array([0.0, 0.0, BLOCK_Z])

        adr = self.block_qpos_adr
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.data.qvel[self.block_dof_adr : self.block_dof_adr + 6] = 0.0
        return fwd, lat, yaw

    def set_targets(self, targets, alpha=1.0):
        """Blend the actuator setpoints towards ``targets`` (alpha=1 -> direct)."""
        for name, value in targets.items():
            idx = self.act_id[name]
            lo, hi = JOINT_LIMITS[name]
            desired = clamp(value, lo, hi)
            self.data.ctrl[idx] += alpha * (desired - self.data.ctrl[idx])

    def set_qpos(self, targets):
        for name, value in targets.items():
            self.data.qpos[self.qpos_adr[name]] = value
        mujoco.mj_forward(self.model, self.data)

    def wrist_position(self):
        return self.data.xpos[self.wrist_body].copy()


def ee_command_to_world(fwd, lat, up):
    """Cartesian teleop command -> world position of the wrist centre.

    ``lat`` no longer places the wrist laterally: it sets ``shoulder_pan``, which
    swings the whole arm plane, so the wrist ends up ``fwd`` away from the pan
    axis along that rotated direction.
    """
    pan = clamp(-PAN_PER_LAT * lat, *JOINT_LIMITS["shoulder_pan"])
    base = np.array([PAN_AXIS_XY[0], PAN_AXIS_XY[1], 0.0])
    heading = FORWARD_WORLD * math.cos(pan) - LEFT_WORLD * math.sin(pan)
    return base + heading * fwd + np.array([0.0, 0.0, up])


def run_selftest(scene_path):
    """Headless check that the IK targets and the simulated FK agree."""
    sim = SO100Sim(scene_path)
    print(f"model: {scene_path}")
    print(f"  nq={sim.model.nq} nu={sim.model.nu}")

    samples = [
        (HOME_FWD, 0.0, HOME_UP, 0.0),
        (0.22, 0.06, 0.20, -0.3),
        (0.16, -0.10, 0.28, 0.4),
        (0.28, 0.0, 0.12, -0.8),
    ]
    worst = 0.0
    for fwd, lat, up, pitch in samples:
        targets = ee_to_joints(fwd, lat, up, pitch, 0.0)
        targets["gripper"] = GRIPPER_OPEN
        sim.reset()
        sim.set_qpos(targets)

        want = ee_command_to_world(fwd, lat, up)
        got = sim.wrist_position()
        err = float(np.linalg.norm(want - got))
        worst = max(worst, err)

        pitch_fk = targets["shoulder_lift"] - targets["elbow_flex"] - targets["wrist_flex"]
        print(
            f"  cmd fwd={fwd:.3f} lat={lat:+.3f} up={up:.3f} pitch={pitch:+.2f}"
            f" | wrist err={err * 1000:6.2f} mm"
            f" | pitch err={math.degrees(pitch_fk - pitch):+6.2f} deg"
        )

    print(f"  worst wrist position error: {worst * 1000:.2f} mm")
    ok = worst < 2e-3
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_viewer(scene_path, device, side, control_hz, randomise_block=True, seed=None):
    sim = SO100Sim(scene_path, rng=np.random.default_rng(seed))
    placement = sim.reset(randomise_block=randomise_block)
    if placement is not None:
        fwd, lat, yaw = placement
        print(
            f"block placed at forward={fwd:.3f} m lateral={lat:+.3f} m "
            f"yaw={math.degrees(yaw):+.0f} deg"
        )
    elif randomise_block:
        print(f"[warn] no free-floating '{BLOCK_JOINT}' joint in {scene_path}; block not moved.")
    target = EETarget()
    source = make_source(device, side)
    print(f"teleop source: {source.name}")
    marker = "Joy-Con (right) controls" if source.name.startswith("joycon") else "Keyboard controls"
    print(__doc__.split(marker, 1)[1])

    key_callback = getattr(source, "key_callback", None)
    control_dt = 1.0 / control_hz
    next_control = time.perf_counter()

    with mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
    ) as viewer:
        sim_start = time.perf_counter()
        try:
            while viewer.is_running():
                if time.perf_counter() >= next_control:
                    next_control += control_dt
                    source.apply(target)
                    sim.set_targets(target.joint_targets(), alpha=0.35)

                mujoco.mj_step(sim.model, sim.data)
                viewer.sync()

                # keep the simulation close to wall-clock time
                lag = (sim_start + sim.data.time) - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            source.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description="SO100 MuJoCo Joy-Con teleoperation")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="MJCF scene to load")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "joycon", "keyboard"],
        help="teleop input device (auto: Joy-Con if present, else keyboard)",
    )
    parser.add_argument("--side", default="right", choices=["right", "left"], help="Joy-Con side")
    parser.add_argument("--control-hz", type=float, default=100.0, help="teleop update rate")
    parser.add_argument(
        "--fixed-block", action="store_true",
        help="keep the block at its scene position instead of sampling a random one",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed for the random block placement"
    )
    parser.add_argument("--selftest", action="store_true", help="headless IK/FK check, no viewer")
    args = parser.parse_args()

    if args.selftest:
        return run_selftest(args.scene)
    return run_viewer(
        args.scene, args.device, args.side, args.control_hz,
        randomise_block=not args.fixed_block, seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
