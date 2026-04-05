# Code block 0
# Get the pose of the red cube
red_cube_pose = get_object_pose("red cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Move to a position above the cube to approach it with the gripper
goto_pose(grasp_position + np.array([0.0, 0.0, 0.1]), grasp_quaternion_wxyz)

# Close the gripper to pick up the red cube
close_gripper()

# Lift the red cube by adjusting the z-axis position
lift_position = grasp_position + np.array([0.0, 0.0, 0.2])
goto_pose(lift_position, grasp_quaternion_wxyz)

# Open the gripper to release the cube
open_gripper()