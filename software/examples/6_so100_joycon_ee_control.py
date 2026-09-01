#!/usr/bin/env python3
"""
Joy-Con end-effector teleoperation for a real SO100/SO101 arm.

The Joy-Con mapping is ported from ``simulation/mujoco/so100_joycon_mujoco.py``,
which in turn follows leisaac's ``SO101JoyConEE`` device: the stick drives the
end effector along the direction the gripper faces, the lateral offset drives
``shoulder_pan`` linearly, and tilting the Joy-Con sets the gripper pitch and
wrist roll.

Unlike the simulation this file talks to real motors, so it keeps the joint
conventions of the original hardware example: the analytic 2-link IK returns
degrees, ``shoulder_lift`` is measured as ``90 - joint2`` and ``elbow_flex`` as
``joint3 - 90``, and the wrist compensates both. Joint commands are therefore in
degrees (lerobot's ``use_degrees=True`` default) and the gripper in percent.

The arm is driven by P control towards the Cartesian target, so the target can
move faster than the arm without the motors being commanded to jump.

Joy-Con (right) controls
------------------------
    stick up/down     : end effector along the gripper's facing direction
    stick left/right  : end effector left / right (drives shoulder_pan)
    R                 : end effector up
    stick press       : end effector down
    X / B             : fine forward / backward
    tilt the Joy-Con  : gripper pitch (nose up/down) and wrist roll
    ZR                : toggle gripper open/close
    HOME              : reset the end effector position
    PLUS              : re-calibrate the Joy-Con IMU and reset the position
    Ctrl+C            : stop
"""

import logging
import math
import time
import traceback

from joyconrobotics import JoyconRobotics

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Joint calibration coefficients - manually edited
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
# ----------------------------------------------------------------------------
JOINT_CALIBRATION = [
    ["shoulder_pan", 6.0, 1.0],  # Joint 1: zero position offset, scale factor
    ["shoulder_lift", 2.0, 0.97],  # Joint 2: zero position offset, scale factor
    ["elbow_flex", 0.0, 1.05],  # Joint 3: zero position offset, scale factor
    ["wrist_flex", 0.0, 0.94],  # Joint 4: zero position offset, scale factor
    ["wrist_roll", 0.0, 0.5],  # Joint 5: zero position offset, scale factor
    ["gripper", 0.0, 1.0],  # Joint 6: zero position offset, scale factor
]

# ----------------------------------------------------------------------------
# Arm geometry (metres). Both lengths are measured from the shoulder-pitch axis,
# which is the frame the IK below works in.
# ----------------------------------------------------------------------------
L1 = 0.1159  # upper arm
L2 = 0.1350  # lower arm

# ----------------------------------------------------------------------------
# Joy-Con -> joint mapping, ported from leisaac's ``SO101JoyConEE``
# ----------------------------------------------------------------------------
# leisaac emits motor-space degrees which are rescaled into the follower's joint
# limits; the constants below fold both steps together, so the same Joy-Con
# motion produces the same joint angle as it does in Isaac and in the MuJoCo
# simulation.
#
#   shoulder_pan  = 300 deg/m   * lateral, motor (-100,100) -> (-110,110)
#   gripper pitch = 300 deg/rad * pitch - 20 deg bias, scale 190/200
#   wrist_roll    =  50 deg/rad * roll,   motor (-100,100) -> (-160,160)
PAN_PER_LAT = 300.0 * 1.1  # deg of shoulder_pan per metre of lateral offset
PITCH_GAIN = math.radians(300.0 * 0.95)  # rad of gripper pitch per rad of Joy-Con tilt
PITCH_OFFSET = math.radians(20.0 * 0.95)  # nose-down bias with the Joy-Con held level
ROLL_GAIN = math.radians(50.0 * 1.6)  # rad of wrist_roll per rad of Joy-Con roll

# Gripper command in percent (lerobot normalises the gripper to 0..100).
GRIPPER_OPEN = 60.0
GRIPPER_CLOSED = 0.0

# Stick handling. The centre values are raw ADC counts and differ between
# Joy-Cons; read yours with ``joycon_test_read_CN.py`` if the arm drifts with the
# stick released.
STICK_V_CENTER = 1800
STICK_H_CENTER = 2000
STICK_DEADZONE = 300
STICK_RANGE = 1000
STICK_SPEED = 0.0008  # metres per control tick at full stick deflection
FINE_STEP = 0.001  # metres per tick for the X / B buttons
DOF_SPEED = [2, 2, 2, 1, 1, 1]

# Direction of the horizontal stick. The simulation negates both this and the
# shoulder_pan sign to match its scene; on the real arm both stay positive,
# which is the direction the original hardware example used.
STICK_LAT_SIGN = 1.0

# Home end-effector pose, measured from the shoulder-pitch axis.
HOME_FWD = 0.1629
HOME_LAT = 0.0
HOME_UP = 0.1131
HOME_PITCH = -PITCH_OFFSET  # gripper pitch, rad, 0 = horizontal, + = nose up
HOME_ROLL = 0.0

# Cartesian workspace clamp (metres / radians), keeps the IK inside a sane
# region. The arm reaches L1 + L2 = 0.251 m, and the IK scales anything beyond
# that back onto the boundary.
FWD_RANGE = (0.03, 0.31)
LAT_RANGE = (-0.28, 0.28)
UP_RANGE = (-0.10, 0.30)
PITCH_RANGE = (math.radians(-95.0), math.radians(95.0))
ROLL_RANGE = (math.radians(-160.0), math.radians(160.0))

# Joint clamps in degrees, derived from the SO101 calibration travel.
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
}


def clamp(value, low, high):
    return max(low, min(high, value))


def apply_joint_calibration(joint_name, raw_position):
    """
    Apply joint calibration coefficients

    Args:
        joint_name: joint name
        raw_position: raw position value

    Returns:
        calibrated_position: calibrated position value
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # zero position offset
            scale = joint_cal[2]  # scale factor
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # if no calibration coefficient found, return original value


def inverse_kinematics(x, y, l1=L1, l2=L2):
    """
    Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets

    Parameters:
        x: End effector x coordinate
        y: End effector y coordinate
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)

    Returns:
        joint2, joint3: Joint angles in degrees as used by the follower
    """
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

    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(clamp(cos_theta2, -1.0, 1.0))

    # Calculate theta1 (shoulder angle)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    # Convert theta1 and theta2 to joint2 and joint3 angles
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset

    # Ensure angles are within URDF limits
    joint2 = clamp(joint2, -0.1, 3.45)
    joint3 = clamp(joint3, -0.2, math.pi)

    # Convert from radians to degrees
    joint2_deg = 90 - math.degrees(joint2)
    joint3_deg = math.degrees(joint3) - 90

    return joint2_deg, joint3_deg


def ee_to_joint_targets(fwd, lat, up, pitch, roll):
    """Map a Cartesian end-effector command to the six SO100 joint targets.

    ``fwd`` / ``lat`` / ``up`` are metres measured from the shoulder-pitch axis;
    ``pitch`` and ``roll`` are radians, ``pitch`` being the gripper pitch with
    0 = horizontal and + = nose up.

    Like leisaac, the lateral offset drives ``shoulder_pan`` linearly instead of
    pointing the arm at a Cartesian target, so the IK sees ``fwd`` alone as the
    reach.
    """
    pan = clamp(lat * PAN_PER_LAT, *JOINT_LIMITS_DEG["shoulder_pan"])

    shoulder_lift, elbow_flex = inverse_kinematics(fwd, up)

    # In this joint convention the wrist compensates the shoulder/elbow pair, and
    # a nose-up gripper pitch bends it the other way.
    wrist_flex = clamp(
        -shoulder_lift - elbow_flex - math.degrees(pitch), *JOINT_LIMITS_DEG["wrist_flex"]
    )
    wrist_roll = clamp(math.degrees(roll), *JOINT_LIMITS_DEG["wrist_roll"])

    return {
        "shoulder_pan": pan,
        "shoulder_lift": shoulder_lift,
        "elbow_flex": elbow_flex,
        "wrist_flex": wrist_flex,
        "wrist_roll": wrist_roll,
    }


class EETarget:
    """Mutable Cartesian end-effector target."""

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
        self.roll = clamp(self.roll, *ROLL_RANGE)

    def joint_targets(self):
        targets = ee_to_joint_targets(self.fwd, self.lat, self.up, self.pitch, self.roll)
        targets["gripper"] = GRIPPER_OPEN if self.gripper_open else GRIPPER_CLOSED
        return targets


class FixedAxesJoyconRobotics(JoyconRobotics):
    """Stick mapping ported from ``so100_joycon_mujoco.py``.

    The vertical stick moves the end effector along the direction the gripper
    currently faces, so it drives forward and height together; the horizontal
    stick drives the lateral offset on its own.
    """

    # Gripper pitch (rad) of the last command, written back by
    # ``JoyconTeleop.apply()``. leisaac steers by the way the *Joy-Con* points;
    # this follows the arm instead, so forward always means "further along the
    # way the jaws point".
    gripper_pitch = HOME_PITCH

    def common_update(self):
        is_right = self.joycon.is_right()

        # unit vector of the gripper in the arm's (forward, up) plane
        pointing_fwd = math.cos(self.gripper_pitch)
        pointing_up = math.sin(self.gripper_pitch)

        stick_v = (
            self.joycon.get_stick_right_vertical()
            if is_right
            else self.joycon.get_stick_left_vertical()
        )
        if abs(stick_v - STICK_V_CENTER) > STICK_DEADZONE:
            delta_v = STICK_SPEED * (stick_v - STICK_V_CENTER) / STICK_RANGE
            self.position[0] += (
                delta_v * self.dof_speed[0] * self.direction_reverse[0] * pointing_fwd
            )
            # leisaac indexes dof_speed / direction_reverse with 1 here
            self.position[2] += (
                delta_v * self.dof_speed[1] * self.direction_reverse[1] * pointing_up
            )

        stick_h = (
            self.joycon.get_stick_right_horizontal()
            if is_right
            else self.joycon.get_stick_left_horizontal()
        )
        if abs(stick_h - STICK_H_CENTER) > STICK_DEADZONE:
            self.position[1] += (
                STICK_SPEED * (stick_h - STICK_H_CENTER) / STICK_RANGE
                * self.dof_speed[1] * self.direction_reverse[1] * STICK_LAT_SIGN
            )

        up = self.joycon.get_button_r() if is_right else self.joycon.get_button_l()
        if up == 1:
            self.position[2] += STICK_SPEED * self.dof_speed[2] * self.direction_reverse[2]
        down = (
            self.joycon.get_button_r_stick() if is_right else self.joycon.get_button_l_stick()
        )
        if down == 1:
            self.position[2] -= STICK_SPEED * self.dof_speed[2] * self.direction_reverse[2]

        fine_fwd = self.joycon.get_button_x() if is_right else self.joycon.get_button_up()
        fine_back = self.joycon.get_button_b() if is_right else self.joycon.get_button_down()
        if fine_fwd == 1:
            self.position[0] += FINE_STEP * self.dof_speed[0]
        elif fine_back == 1:
            self.position[0] -= FINE_STEP * self.dof_speed[0]

        home = self.joycon.get_button_home() if is_right else self.joycon.get_button_capture()
        if home == 1:
            self.position = self.offset_position_m.copy()

        self.button_control = 0
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
            # leisaac maps PLUS to an IMU re-calibration; A / Y drive the episode
            # recorder there, which has no counterpart here.
            recalibrate = (is_right and event_type == "plus") or (
                not is_right and event_type == "minus"
            )
            if recalibrate and status == 1:
                self.position = self.offset_position_m.copy()
                self.reset_joycon()
                self.button_control = 8

        return self.position, self.gripper_state, self.button_control


class JoyconTeleop:
    """Joy-Con teleop using leisaac's ``SO101JoyConEE`` mapping."""

    def __init__(self, side="right"):
        self._joycon = FixedAxesJoyconRobotics(side, dof_speed=DOF_SPEED)
        self.name = f"joycon-{side}"

    def apply(self, target: EETarget):
        pose, gripper, _ = self._joycon.get_control()
        dfwd, dlat, dup, roll, pitch, _yaw = pose

        target.fwd = HOME_FWD + dfwd
        target.lat = HOME_LAT + dlat
        target.up = HOME_UP + dup
        # leisaac: pitch_deg = -pitch * 300 + 20, folded into the wrist by
        # ee_to_joint_targets().
        target.pitch = pitch * PITCH_GAIN - PITCH_OFFSET
        target.roll = roll * ROLL_GAIN
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


def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    Use P control to slowly move robot to zero position

    Args:
        robot: robot instance
        duration: time to move to zero position (seconds)
        kp: proportional gain
    """
    print("Using P control to slowly move robot to zero position...")

    # Zero position targets
    zero_positions = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    }

    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    print(
        f"Will use P control to move to zero position in {duration} seconds, control frequency: {control_freq}Hz, proportional gain: {kp}"
    )

    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                # Apply calibration coefficients
                current_positions[motor_name] = apply_joint_calibration(motor_name, value)

        # P control calculation
        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos

                # P control: output = Kp * error
                control_output = kp * error

                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)

        # Show progress
        if step % (control_freq // 2) == 0:  # Show progress every 0.5 seconds
            progress = (step / total_steps) * 100
            print(f"Moving to zero position progress: {progress:.1f}%")

        time.sleep(step_time)

    print("Robot has moved to zero position")


def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    Use P control to return to start position

    Args:
        robot: robot instance
        start_positions: start joint position dictionary
        kp: proportional gain
        control_freq: control frequency (Hz)
    """
    print("Returning to start position...")

    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds

    for step in range(max_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                current_positions[motor_name] = value  # Don't apply calibration coefficients

        # P control calculation
        robot_action = {}
        total_error = 0
        for joint_name, target_pos in start_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)

                # P control: output = Kp * error
                control_output = kp * error

                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)

        # Check if reached start position
        if total_error < 2.0:  # If total error is less than 2 degrees, consider reached
            print("Returned to start position")
            break

        time.sleep(control_period)

    print("Return to start position completed")


def p_control_loop(robot, teleop, target, kp=0.5, control_freq=50, status_hz=2.0):
    """
    P control loop

    Args:
        robot: robot instance
        teleop: teleop source writing into ``target``
        target: Cartesian end-effector target
        kp: proportional gain
        control_freq: control frequency (Hz)
        status_hz: how often the current command is printed
    """
    control_period = 1.0 / control_freq
    status_period = 1.0 / status_hz if status_hz > 0 else None
    next_status = time.perf_counter()

    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")

    while True:
        try:
            # Read the Joy-Con and update the Cartesian target
            teleop.apply(target)
            target_positions = target.joint_targets()

            # Get current robot state
            current_obs = robot.get_observation()

            # Extract current joint positions
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith(".pos"):
                    motor_name = key.removesuffix(".pos")
                    # Apply calibration coefficients
                    current_positions[motor_name] = apply_joint_calibration(motor_name, value)

            # P control calculation
            robot_action = {}
            for joint_name, target_pos in target_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos

                    # P control: output = Kp * error
                    control_output = kp * error

                    # Convert control output to position command
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position

            # Send action to robot
            if robot_action:
                robot.send_action(robot_action)

            if status_period is not None and time.perf_counter() >= next_status:
                next_status += status_period
                print(
                    f"fwd={target.fwd:.3f} lat={target.lat:+.3f} up={target.up:.3f}"
                    f" pitch={math.degrees(target.pitch):+6.1f}"
                    f" roll={math.degrees(target.roll):+6.1f}"
                    f" gripper={'open' if target.gripper_open else 'closed'}"
                )

            time.sleep(control_period)

        except KeyboardInterrupt:
            print("User interrupted program")
            break
        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            break


def main():
    """Main function"""
    print("SO100/SO101 Joy-Con End-Effector Teleoperation (P Control)")
    print("=" * 50)

    teleop = None
    robot = None
    try:
        # Import necessary modules
        from lerobot.robots.so_follower.so_follower import SO100Follower
        from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig

        # Get port
        port = input("Please enter the USB port for SO100 robot (e.g., /dev/ttyACM0): ").strip()

        # If directly press Enter, use default port
        if not port:
            port = "/dev/ttyACM0"
            print(f"Using default port: {port}")
        else:
            print(f"Connecting to port: {port}")

        # Configure robot
        robot_config = SO100FollowerConfig(port=port)
        robot = SO100Follower(robot_config)

        # Connect devices
        robot.connect()
        print("Device connection successful!")

        # Ask whether to recalibrate
        while True:
            calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
            if calibrate_choice in ["y", "yes"]:
                print("Starting recalibration...")
                robot.calibrate()
                print("Calibration completed!")
                break
            elif calibrate_choice in ["n", "no"]:
                print("Using previous calibration file")
                break
            else:
                print("Please enter y or n")

        # Read initial joint angles
        print("Reading initial joint angles...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                start_positions[motor_name] = int(value)  # Don't apply calibration coefficients

        print("Initial joint angles:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}")

        # Move to zero position
        move_to_zero_position(robot, duration=3.0)

        # Connect the Joy-Con only once the arm is in a known pose
        teleop = JoyconTeleop("right")
        print(f"teleop source: {teleop.name}")
        print(__doc__.split("Joy-Con (right) controls", 1)[1])

        target = EETarget()
        print(
            f"Initialize end effector position: fwd={target.fwd:.4f} "
            f"lat={target.lat:+.4f} up={target.up:.4f}"
        )
        print("=" * 50)
        print("Note: Robot will continuously move to target positions")

        # Start P control loop
        p_control_loop(robot, teleop, target, kp=0.5, control_freq=50)

    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Whether the robot is properly connected")
        print("2. Whether the USB port is correct")
        print("3. Whether you have sufficient permissions to access USB devices")
        print("4. Whether the robot is properly configured")
    finally:
        if teleop is not None:
            teleop.close()
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
        print("Program ended")


if __name__ == "__main__":
    main()
