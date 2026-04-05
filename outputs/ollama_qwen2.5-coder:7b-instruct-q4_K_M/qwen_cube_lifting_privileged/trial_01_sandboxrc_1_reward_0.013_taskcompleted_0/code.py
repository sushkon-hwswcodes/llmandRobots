# Code block 0
# Sample a grasp pose for the red cube
cube_grasp_pos, cube_grasp_quat = sample_grasp_pose("red cube")

# Move the robot to above the cube
goto_pose(cube_grasp_pos + np.array([0, 0, 0.1]), cube_grasp_quat)

# Close the gripper to grasp the cube
close_gripper()

# Lift the cube up by a small amount to ensure it's picked up
goto_pose(cube_grasp_pos + np.array([0, 0, 0.2]), cube_grasp_quat)