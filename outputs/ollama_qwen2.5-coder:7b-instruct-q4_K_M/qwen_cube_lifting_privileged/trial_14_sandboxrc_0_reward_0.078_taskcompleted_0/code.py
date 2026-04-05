# Code block 0
import numpy as np

# Get the position and quaternion of the red cube
red_cube_position, red_cube_quaternion, _ = get_object_pose("red cube", return_bbox_extent=True)

# Sample a grasp pose for the red cube
grasp_pose_position, grasp_pose_quaternion = sample_grasp_pose("red cube")

# Approach the cube with the gripper
z_approach = 0.15
goto_pose(grasp_pose_position + np.array([0, 0, z_approach]), grasp_pose_quaternion)

# Close the gripper to pick up the cube
close_gripper()

# Lift the cube up
lift_position = red_cube_position + np.array([0, 0, 0.1])  # Adjust the lift distance as needed
goto_pose(lift_position, grasp_pose_quaternion)