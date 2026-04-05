# Code block 0
import numpy as np

# Get the pose of the red cube
cube_pose, cube_quaternion_wxyz, bbox_extent = get_object_pose("red cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Move the arm to the grasp position
goto_pose(grasp_position + np.array([0, 0, 0.1]), grasp_quaternion_wxyz)

# Close the gripper to pick up the cube
close_gripper()

# Lift the cube
new_position = get_object_pose("base of environment")[0] + np.array([0, 0, 0.2])
goto_pose(new_position, get_grasped_object_quaternion())

# Open the gripper to release the cube
open_gripper()