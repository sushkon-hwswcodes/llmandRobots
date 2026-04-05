# Code block 0
import numpy as np

# Get the pose of the red cube
cube_position, cube_quaternion, _ = get_object_pose('red cube', False)

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion = sample_grasp_pose('red cube')

# Move the robot to above the cube with an approach distance
goto_pose(grasp_position, grasp_quaternion, z_approach=0.1)

# Close the gripper to pick up the cube
close_gripper()

# Lift the red cube by moving its position slightly
lift_position = cube_position + np.array([0, 0, 0.1])
goto_pose(lift_position, grasp_quaternion)

# Open the gripper to release the cube (if needed)
open_gripper()