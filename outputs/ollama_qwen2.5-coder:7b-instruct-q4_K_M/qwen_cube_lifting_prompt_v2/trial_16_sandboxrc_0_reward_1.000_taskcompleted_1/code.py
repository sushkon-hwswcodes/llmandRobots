# Code block 0
import numpy as np

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Open the gripper
open_gripper()

# Go to approach position
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Go to grasp position
goto_pose(grasp_pos, grasp_quat)

# Close the gripper
close_gripper()

# Determine the lift position
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1

# Go to lift position
goto_pose(lift_pos, grasp_quat)