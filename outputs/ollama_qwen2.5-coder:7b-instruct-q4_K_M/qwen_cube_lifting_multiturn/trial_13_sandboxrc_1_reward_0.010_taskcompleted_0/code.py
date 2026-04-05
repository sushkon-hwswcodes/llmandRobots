# Code block 0
# Define the object we need to pick up and its target position
target_object = "red_cube"
target_position = np.array([0.5, 0.1, 0.8])  # Example target position above the red cube

# Sample a grasp pose for the red cube
grasp_pose = sample_grasp_pose(target_object)
grasp_position = grasp_pose[0]
grasp_quaternion = grasp_pose[1]

# Goto the initial approach position with Z-approach
goto_pose(grasp_position, grasp_quaternion, z_approach=0.05)

# Close the gripper to pick up the object
close_gripper()

# Move to the target position above the red cube
goto_pose(target_position, grasp_quaternion, z_approach=0.05)

# Open the gripper to release the object
open_gripper()