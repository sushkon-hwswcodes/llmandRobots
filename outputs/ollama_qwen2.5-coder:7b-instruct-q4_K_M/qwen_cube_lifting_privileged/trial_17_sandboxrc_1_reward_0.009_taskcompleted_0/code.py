# Code block 0
# Get the pose of the red cube
red_cube_pose, _, _ = get_object_pose("red cube")
# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Approach to the cube a bit above it to ensure stable grasping
goto_pose(grasp_position + np.array([0, 0, 0.1]), grasp_quaternion_wxyz)
# Close the gripper to pick up the cube
close_gripper()
# Continue moving above the red cube while holding it
goto_pose(grasp_position + np.array([0, 0, 0.3]), grasp_quaternion_wxyz)

print("Red cube picked up and lifted")