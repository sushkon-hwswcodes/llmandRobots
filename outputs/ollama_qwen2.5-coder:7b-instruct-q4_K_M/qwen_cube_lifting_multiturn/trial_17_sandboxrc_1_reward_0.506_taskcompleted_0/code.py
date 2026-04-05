# Code block 0
# Sample a grasp pose for the red cube
red_cube_grasp_pose = sample_grasp_pose("red cube")

# Go to the sampled grasp pose with a slight approach
goto_pose(red_cube_grasp_pose[0], red_cube_grasp_pose[1], z_approach=0.1)

# Close the gripper to pick up the red cube
close_gripper()

# Lift the red cube by adjusting the position
lift_position = red_cube_grasp_pose[0] + numpy.array([0, 0, 0.2])
goto_pose(lift_position, red_cube_grasp_pose[1], z_approach=0)

# Open the gripper to release the red cube
open_gripper()