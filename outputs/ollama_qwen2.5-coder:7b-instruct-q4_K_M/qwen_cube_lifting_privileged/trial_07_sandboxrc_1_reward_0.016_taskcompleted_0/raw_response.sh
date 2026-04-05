# Get the pose of the red cube
red_cube_pos, _, _ = get_object_pose('red cube')

# Sample a grasp pose for the red cube
grasp_pos, _ = sample_grasp_pose('red cube')

# Move the robot to above the red cube
goto_pose(grasp_pos, np.array([0, 0, 1, 0]), z_approach=0.1)

# Close the gripper to pick up the red cube
close_gripper()

# Lift the red cube
goto_pose(red_cube_pos, np.array([0, 0, 1, 0]), z_approach=0.3)