# Code block 0
import numpy as np

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Open the gripper to prepare for grasping
open_gripper()

# Move the robot to an approach position above the red cube
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Move the robot to the exact grasp pose
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to grasp the red cube
close_gripper()

# Lift the red cube by moving it upwards
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)