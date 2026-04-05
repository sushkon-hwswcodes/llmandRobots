# Code block 0
import numpy as np

approach_pos, _ = sample_grasp_pose("red_cube")
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")
goto_pose(approach_pos, grasp_quat, z_approach=0.1)
goto_pose(grasp_pos, grasp_quat)
close_gripper()
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)