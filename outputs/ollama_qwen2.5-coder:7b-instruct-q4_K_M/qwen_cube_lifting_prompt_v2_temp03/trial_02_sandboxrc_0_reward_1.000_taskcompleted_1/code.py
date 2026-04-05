# Code block 0
import numpy as np

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Open the gripper to prepare for grasping
open_gripper()

# Approach the cube with a small z offset for better control
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Move to the exact grasp position
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to grasp the cube
close_gripper()

# Lift the cube by increasing the Z coordinate slightly
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)