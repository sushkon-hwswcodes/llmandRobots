# Code block 0
import numpy as np

# Get the pose of the red cube
red_cube_pose = get_object_pose("red cube")
red_cube_position, red_cube_quaternion, _ = red_cube_pose

# Sample a grasp pose for the red cube
grasp_pose = sample_grasp_pose("red cube")
grasp_position, grasp_quaternion = grasp_pose

# Move to an approach position above the red cube
approach_position = np.copy(red_cube_position)
approach_position[2] += 0.1  # Adjust the z-axis for approach distance
goto_pose(approach_position, grasp_quaternion, z_approach=0.1)

# Close the gripper to pick up the red cube
close_gripper()

# Move upward while holding the object
lift_position = np.copy(grasp_position)
lift_position[2] += 0.3  # Adjust the lifting height as needed
goto_pose(lift_position, grasp_quaternion)

# Open the gripper after lifting to release the red cube
open_gripper()