# Code block 0
# Get the pose of the red cube object
cube_pose, _, _ = get_object_pose("red cube")

# Sample a grasp pose for the red cube object
grasp_pos, grasp_quat = sample_grasp_pose("red cube")

# Go to the grasp position with an approach motion
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Close the gripper fully to pick up the cube
close_gripper()

# Raise the red cube by adjusting the Z-axis in the original object pose
raise_grasp_pos = cube_pose + np.array([0, 0, 0.2])  # Adjust the height as needed
goto_pose(raise_grasp_pos, grasp_quat)

# Open the gripper to release the cube
open_gripper()