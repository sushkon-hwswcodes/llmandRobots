import numpy as np

# Get the pose of the red cube
cube_pos, cube_quat = get_object_pose("red_cube")

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Move to an approach position above the cube
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Move to the actual grasp position
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to pick up the red cube
close_gripper()

# Move the cube slightly higher in case there are obstacles below it
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)