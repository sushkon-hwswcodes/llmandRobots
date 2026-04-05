# Code block 0
# Get the pose of the red cube
cube_position, _, _ = get_object_pose("red cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion = sample_grasp_pose("red cube")

# Approach to the cube with some z-offset
goto_pose(grasp_position, grasp_quaternion, z_approach=0.1)

# Close the gripper to pick up the cube
close_gripper()

# Move above the red cube
goto_pose(cube_position + np.array([0, 0, 0.2]), grasp_quaternion)