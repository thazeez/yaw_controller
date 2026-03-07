# Yaw Controller Experiments for PX4 Offboard Control

This repository contains scripts used to test yaw-heading control and offboard velocity commands for a PX4-based drone using ROS 2.

The experiments were conducted using two terminals:

- **Terminal 1:** Run the offboard relay  
- **Terminal 2:** Run the yaw controller  

---

# Repository Structure

The repository contains the following main folders:

**scripts/**  
Contains all control and relay scripts.

Inside this folder:

- **scripts/controller/**
  - `YawController.py`
  - `YawController_OptiWorld_Clean.py`
  - `YawPDController.py`

- **scripts/offboard_relay/**
  - `offboard_vel_relay_body3_fullquat_withyaw.py`

**plotting/**  
Contains scripts used to generate plots from the experiment data.

**csv/**  
Contains logged experimental data.

**Figures/**  
Contains generated plots from the experiments.

---

# Running the Experiments

Two terminals are required.

## Terminal 1 — Offboard Relay

Run the offboard relay script:

```bash
python3 scripts/offboard_relay/offboard_vel_relay_body3_fullquat_withyaw.py
```

This script handles communication with PX4 and publishes velocity commands.

---

## Terminal 2 — Run the Controller

Run the yaw controller depending on the experiment.

---

# Controller Descriptions

## YawController_OptiWorld_Clean.py

Used for **Tests 1 – 4**.

This controller receives a **commanded yaw angle from the user** and commands the drone to rotate to the desired heading.

The drone successfully tracks the commanded yaw angle.

---

## YawPDController.py

Used for **Tests 5 – 7**.

This controller receives a **desired world position** and computes the heading error between the drone and the target position. The controller then commands the drone to yaw toward the goal and move forward.

### Observed Behavior

The drone physically yawed in the opposite direction and moved in the opposite direction.

However, from the experimental figures it can be seen that the **heading was tracked correctly**, but the drone moved in the opposite direction.

---

## Sign Flip Experiments

**Tests 8 – 9**

The yaw sign was flipped in the controller.

Results:

- The drone still moved in the opposite direction
- The heading was also tracked in the opposite direction

---

# Experimental Data

Experimental logs are stored in the **csv/** folder.

Generated figures from the experiments are stored in the **Figures/** folder.

Plotting scripts used to generate these figures are located in the **plotting/** folder.

---

# Author

Taiwo Hazeez  
PhD Student – Mechanical Engineering  
Florida Atlantic University
