# Code block 0
import numpy as np

# Get the pose of the red cube
red_cube_pose, _, _ = get_object_pose("red_cube")

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Approach and grab the red cube
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
goto_pose(grasp_pos, grasp_quat)
close_gripper()

# Lift the red cube above the table
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1  # Adjust the height as needed
goto_pose(lift_pos, grasp_quat)