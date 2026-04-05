# Code block 0
# Get the pose of the red cube object
red_cube_pose, _quaternion_wxyz, bbox_extent = get_object_pose('red cube', return_bbox_extent=True)

# Sample a grasp pose for the red cube object
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose('red cube')

# Move the robot to above the red cube
goto_pose(grasp_position + [0.2], grasp_quaternion_wxyz)

# Close the gripper to pick up the red cube
close_gripper()

# Move the robot's wrist slightly down to ensure stable grasping of the red cube
goto_pose([grasp_position[0], grasp_position[1], grasp_position[2] - 0.05], grasp_quaternion_wxyz)

# Lift the red cube up (e.g., lift it to a height of 0.3 meters from its initial position)
goto_pose([grasp_position[0], grasp_position[1], grasp_position[2] + 0.3], grasp_quaternion_wxyz)