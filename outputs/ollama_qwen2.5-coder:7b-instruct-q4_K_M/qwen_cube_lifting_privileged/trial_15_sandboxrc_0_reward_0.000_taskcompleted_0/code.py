# Code block 0
import numpy as np

# Get the pose of the red cube
red_cube_position, _, _ = get_object_pose("red cube", return_bbox_extent=False)

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Move the robot to above the cube with an approach height of 0.1 meters
goto_pose(grasp_position, grasp_quaternion_wxyz, z_approach=0.1)

# Approach the cube slightly
approach_offset = np.array([0, 0, -0.05])
final_grasp_position = grasp_position + approach_offset
goto_pose(final_grasp_position, grasp_quaternion_wxyz)

# Close the gripper to pick up the red cube
close_gripper()

# Open the gripper above a certain height to avoid collisions
lift_height = np.array([0, 0, 0.1])
final_lift_position = final_grasp_position + lift_height
goto_pose(final_lift_position, grasp_quaternion_wxyz)

# Move the robot to place the red cube somewhere else (e.g., on a table)
place_position = np.array([0.5, 0.2, 0.3])  # Example place position
final_place_position = place_position + lift_height
goto_pose(final_place_position, grasp_quaternion_wxyz)

# Open the gripper to release the red cube
open_gripper()