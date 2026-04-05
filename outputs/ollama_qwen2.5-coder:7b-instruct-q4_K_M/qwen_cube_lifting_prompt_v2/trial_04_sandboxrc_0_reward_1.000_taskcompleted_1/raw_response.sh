import numpy as np

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Move to an approach position above the cube
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Move to the exact grasp位置
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to pick up the red cube
close_gripper()

# Define a lift position above the table (assuming a height of 0.3 meters from the table)
lift_height = 0.3
lift_pos = grasp_pos.copy()
lift_pos[2] += lift_height

# Move the lifted cube to the desired lift position
goto_pose(lift_pos, grasp_quat)