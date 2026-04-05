# Code block 0
import numpy as np
from franka_api import get_object_pose, sample_grasp_pose, goto_pose, open_gripper, close_gripper

# Get the pose of the red cube
cube_position, cube_quaternion_wxyz, _ = get_object_pose("red cube")

# Sample a grasp pose for the red cube
grasp_position, gripper_down_quaternion_wxyz = sample_grasp_pose("red cube")

# Approach and open the gripper to pick up the red cube
goto_pose(graspposition + np.array([0, 0, 0.1]), gripper_down_quaternion_wxyz, z_approach=0.2)
open_gripper()

# Close the gripper to grab the red cube
close_gripper()

# Lift the red cube
lift_position = cube_position.copy()
lift_position[2] += 0.15  # Adjust height for lifting
goto_pose(lift_position, gripper_down_quaternion_wxyz)

# Open the gripper after lifting
open_gripper()