# Code block 0
import numpy as np

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Open the gripper
open_gripper()

# Approach to the cube with a small z offset
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Move to the exact grasp pose
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to grasp the cube
close_gripper()

# Lift the cube above the table by increasing the Z position slightly
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)