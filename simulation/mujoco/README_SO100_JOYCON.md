# SO100 in MuJoCo — Joy-Con teleoperation

Standalone SO-ARM100 arm for MuJoCo plus a Cartesian teleoperation script driven
by a Nintendo Joy-Con (with a keyboard fallback).

| File | What it is |
|---|---|
| `so100.xml` | the SO100 arm on its own (six actuated joints, meshes from `assets/`) |
| `so100_scene.xml` | floor, lighting, a graspable block, and the home keyframe |
| `so100_joycon_mujoco.py` | the teleoperation script |
| `run_so100_joycon.bat` | Windows launcher (activates the conda env, then runs the script) |
| `requirements_so100.txt` | minimal dependency list |

## 1. Conda environment

```bash
conda create -y -n xlerobot python=3.10
conda activate xlerobot
pip install -r simulation/mujoco/requirements_so100.txt
```

`joyconrobotics` is **not** installed from PyPI — the copy vendored at
`software/joyconrobotics/` is used, and the script puts `software/` on
`sys.path` itself, so nothing else is needed.

## 2. Run

```bash
cd simulation/mujoco
python so100_joycon_mujoco.py                 # Joy-Con if present, else keyboard
python so100_joycon_mujoco.py --device joycon # require a Joy-Con (error out if missing)
python so100_joycon_mujoco.py --device keyboard
python so100_joycon_mujoco.py --side left     # use the left Joy-Con
python so100_joycon_mujoco.py --seed 0        # reproducible block placement
python so100_joycon_mujoco.py --fixed-block   # block stays where the scene puts it
python so100_joycon_mujoco.py --selftest      # headless IK/FK check, no viewer window
```

The block is dropped at a random reachable spot on every start, and the sampled
pose is printed. `--seed` makes that draw repeatable; `--fixed-block` keeps the
scene's own `0.10 -0.20` position.

`--selftest` loads the model, commands a handful of Cartesian poses and compares
the resulting forward kinematics with the command. It should print
`SELFTEST: PASS` with a sub-millimetre error.

### Windows launcher

`run_so100_joycon.bat` does the env activation for you — double-click it, or call
it with the same arguments:

```bat
run_so100_joycon.bat
run_so100_joycon.bat --device keyboard
run_so100_joycon.bat --selftest
```

It searches the usual Miniconda/Anaconda install locations. If yours is somewhere
else, or the env has another name, set them first:

```bat
set CONDA_ROOT=C:\Users\me\miniconda3
set ENV_NAME=my_env
run_so100_joycon.bat
```

## 3. Controls

### Joy-Con (right; the left Joy-Con uses the mirrored buttons)

| Input | Action |
|---|---|
| stick up / down | move along the direction the gripper faces (couples x and z) |
| stick left / right | end effector left / right (drives `shoulder_pan`, sign from `STICK_LAT_SIGN`) |
| `R` | end effector up |
| stick press | end effector down |
| `X` / `B` | fine forward / backward |
| tilt the controller | gripper pitch (nose up/down) and wrist roll |
| `ZR` | toggle gripper open/close |
| `HOME` | reset the end effector position |
| `+` | re-calibrate the IMU and reset the position |

Pair the Joy-Con over Bluetooth **before** starting the script. On Windows the
`hidapi` wheel ships its own DLL, so no extra driver install is needed.

### Keyboard (viewer window must have focus)

| Keys | Action |
|---|---|
| `W` / `S` | forward / backward |
| `A` / `D` | left / right |
| `R` / `F` | up / down |
| `T` / `G` | gripper pitch up / down |
| `Y` / `H` | wrist roll ± |
| `SPACE` | toggle gripper |
| `0` | reset to home |

## 4. How the control works

The teleop target is a Cartesian point `(forward, lateral, height)` measured from
the shoulder-pan axis, plus a gripper pitch and roll. `ee_to_joints()` converts
it to the six joint angles:

* `shoulder_pan = -PAN_PER_LAT * lateral`, a linear drive rather than a
  point-at-the-target `atan2`, so the IK sees `forward` alone as the reach.
* `shoulder_lift`, `elbow_flex` come from the same analytic 2-link IK used by the
  real-robot examples in `software/examples/` (`L1 = 0.1159 m`, `L2 = 0.1350 m`).
  Unlike those examples this version returns **radians**, which are the MuJoCo
  joint values directly — no degree conversion step.
* `wrist_flex = shoulder_lift - elbow_flex - pitch`, because in this model the
  gripper pitch is exactly `shoulder_lift - elbow_flex - wrist_flex`.

### Joy-Con mapping

The mapping is ported from leisaac's `SO101JoyConEE`
(`leisaac/devices/gamepad/joycon_ee_gamepad.py`, the `joycon-ee` device of
`teleop_se3_agent2.py`). leisaac emits motor-space degrees that
`convert_action_from_so101_leader()` rescales from the motor limits into the USD
joint limits before converting to radians; the constants here fold both steps
together, so the same Joy-Con motion yields the same joint angle as in Isaac:

| leisaac | here | value |
|---|---|---|
| `shoulder_pan = y * 300` deg, motor→USD ×1.1 | `PAN_PER_LAT` | `5.760 rad/m` |
| `pitch_deg = -pitch * 300 + 20`, USD ×0.95 | `PITCH_GAIN` / `PITCH_OFFSET` | `4.974 rad/rad`, `0.332 rad` |
| `roll_deg = roll * 50`, motor→USD ×1.6 | `ROLL_GAIN` | `1.396 rad/rad` |
| gripper `60` / `20` motor deg, motor→USD ×1.1 −10 | `GRIPPER_OPEN` / `GRIPPER_CLOSED` | `0.977` / `0.209 rad` |

One deliberate deviation: leisaac multiplies the vertical stick by
`direction_vector`, the unit vector the **Joy-Con** points along. Here the stick
follows the **gripper** instead — `JoyconSource.apply()` writes the clamped
command back to `FixedAxesJoycon.gripper_pitch`, and the stick travels along
`(cos(pitch), sin(pitch))` in the arm's forward/up plane. Forward therefore
always means "further along the way the jaws point", and because the gripper
pitch is `PITCH_GAIN` (≈5×) the Joy-Con tilt, a small tilt steers the stick much
more sharply than it did under leisaac's version.

The IK positions the **wrist centre**, matching the real-robot examples. The
`grasp_centre` site (small red dot) marks where an object ends up between the
fingers.

## 5. Notes on the model

* **Self-collision is off for the arm.** Every arm geom is `contype="1"
  conaffinity="0"`, so links never collide with each other but still collide with
  the floor and scene objects. Without this the base shell and the shoulder
  overlap by design and the pan joint jams.
* **The fingers use the dedicated collision meshes** (`Fixed_Jaw_Collision_*`,
  `Moving_Jaw_Collision_*`) instead of the visual meshes, which makes grasping
  behave sensibly.
* **Actuators** are `position` with `kp=200`, `dampratio=1`, `forcerange=±35`.
  Joint tracking settles inside 1°.
* **To grasp the block, roll the wrist ~90°.** With `wrist_roll = 0` the jaws
  open and close in the arm's vertical plane, so a top-down approach presses a
  finger onto the object instead of straddling it.
* The block is 2 × 2 × 6 cm and deliberately tall: the fingertips reach about
  2 cm past the grasp centre, so a flat object lying on the floor cannot be
  picked from directly above without the fingers hitting the ground.
* **The spawn box is set in code**, in the same arm-centred frame as the teleop
  command:

  | constant | default | meaning |
  |---|---|---|
  | `BLOCK_FWD_RANGE` | `(0.13, 0.23)` | metres forward of the shoulder-pan axis |
  | `BLOCK_LAT_RANGE` | `(-0.11, 0.11)` | metres sideways, + = left |
  | `BLOCK_YAW_RANGE` | `(-pi, pi)` | spawn heading |
  | `BLOCK_Z` | `0.030` | half the block height, so it rests on the floor |

  The arm reaches `SHOULDER_FWD + L1 + L2 = 0.281 m` at shoulder height and less
  near the floor, so widening `BLOCK_FWD_RANGE` much past `0.24` puts the block
  out of reach.

## 6. Troubleshooting

* `Joy-Con unavailable (...); falling back to keyboard control.` — the Joy-Con is
  not paired, or `hidapi` cannot open it. Use `--device joycon` to see the full
  error instead of the fallback.
* If `pip list` inside the env shows unexpected versions, a per-user
  `site-packages` directory may be shadowing the env. Run with
  `PYTHONNOUSERSITE=1` to confirm.
