# Code block 0
# Get the pose of the red cube
cube_position, _, _ = get_object_pose('red cube')

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose('red cube')

# Move to the approach position above the cube
goto_pose(grasp_position, grasp_quaternion_wxyz, z_approach=0.1)

# Close the gripper to grab the cube
close_gripper()

# Lift the cube by moving upwards
lift_position = grasp_position.copy()
lift_position[2] += 0.15  # Assuming we want to lift it higher than the current grasp position
goto_pose(lift_position, grasp_quaternion_wxyz)