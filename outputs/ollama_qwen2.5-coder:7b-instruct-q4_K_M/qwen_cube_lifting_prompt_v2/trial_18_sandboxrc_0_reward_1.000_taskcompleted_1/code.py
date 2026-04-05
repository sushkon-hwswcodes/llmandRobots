# Code block 0
import numpy as np

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Open the gripper
open_gripper()

# Approach to the grasp position with a small vertical offset
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Move to the grasp position without additional vertical offset
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to grasp the red cube
close_gripper()

# Define the lift position by raising the cube vertically by 0.1 meters from the grasp position
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1

# Move the gripper to the lift position while holding the cube
goto_pose(lift_pos, grasp_quat)