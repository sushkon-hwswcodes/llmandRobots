# Code block 0
import numpy as np

# Get the pose of the red cube
cube_pos, cube_quat, _ = get_object_pose("red_cube")

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Move to the approach pose
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Move to the grasp pose
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to grab the cube
close_gripper()

# Lift the cube
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)