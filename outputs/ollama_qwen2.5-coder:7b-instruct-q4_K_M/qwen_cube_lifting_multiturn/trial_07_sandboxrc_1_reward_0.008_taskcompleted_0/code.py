# Code block 0
from franka_gripper import Grasp, Robotiq2FingerGripper
from typing import Tuple
import numpy as np

# Define the object name
object_name = "red_cube"

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose(object_name)

# Goto the sampled grasp pose with a small z_approach for better control
goto_pose(grasp_position, grasp_quaternion_wxyz, z_approach=0.1)

# Close the gripper to pick up the cube
close_gripper()

# Lift the object by changing its Z position while keeping the same orientation
lift_height = 0.2  # Desired height above the original position
lifted_robot_pose = np.array([grasp_position[0], grasp_position[1], grasp_position[2] + lift_height])
goto_pose(lifted_robot_pose, grasp_quaternion_wxyz)

# Open the gripper again to release the cube
open_gripper()