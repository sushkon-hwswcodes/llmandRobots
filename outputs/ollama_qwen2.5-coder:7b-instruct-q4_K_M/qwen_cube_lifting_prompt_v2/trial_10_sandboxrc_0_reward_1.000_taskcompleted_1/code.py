# Code block 0
import numpy as np

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Go to the approach position
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Go to the grasp position
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to grasp the cube
close_gripper()

# Calculate the lift position above the table
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1

# Go to the lift position
goto_pose(lift_pos, grasp_quat)