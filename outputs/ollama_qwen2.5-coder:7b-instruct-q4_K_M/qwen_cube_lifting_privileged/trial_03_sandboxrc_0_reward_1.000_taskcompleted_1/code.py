# Code block 0
# Get the pose of the red cube
cube_position, _, _ = get_object_pose("red_cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red_cube")

# Move the robot's end effector to the grasping position
goto_pose(grasp_position, grasp_quaternion_wxyz)

# Close the gripper to pick up the red cube
close_gripper()

# Lift the red cube by moving the end effector above its current position
lift_position = grasp_position.copy()
lift_position[2] += 0.1  # Move up by 10 cm
goto_pose(lift_position, grasp_quaternion_wxyz)