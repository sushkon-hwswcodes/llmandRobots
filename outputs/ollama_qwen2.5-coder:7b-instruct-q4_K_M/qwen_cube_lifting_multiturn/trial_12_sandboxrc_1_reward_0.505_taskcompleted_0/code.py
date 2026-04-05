# Code block 0
# Sample a grasp pose for the red cube
red_cube_pose = sample_grasp_pose("red cube")

# Move to the sampled grasp pose with a small z approach to ensure we are above the object
goto_pose(red_cube_pose[0], red_cube_pose[1], z_approach=0.05)

# Close the gripper to pick up the red cube
close_gripper()

# Define a lift height
lift_height = 0.1

# Move up after picking up the red cube
final_position = red_cube_pose[0] + np.array([0, 0, lift_height])
goto_pose(final_position, red_cube_pose[1])

# Open the gripper to release the red cube above the current position
open_gripper()