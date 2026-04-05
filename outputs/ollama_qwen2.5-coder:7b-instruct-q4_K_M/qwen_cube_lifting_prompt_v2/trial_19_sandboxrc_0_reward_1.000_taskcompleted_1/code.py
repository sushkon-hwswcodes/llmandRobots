# Code block 0
import numpy as np

# Get the current pose of the red cube
cube_pos, cube_quat, _ = get_object_pose("red_cube", return_bbox_extent=True)

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Move to the approach position above the red cube
approach_pos = grasp_pos.copy()
approach_pos[2] += 0.1
goto_pose(approach_pos, grasp_quat)

# Move to the grasp position and open the gripper
goto_pose(grasp_pos, grasp_quat)
close_gripper()

# Lift the red cube into the air
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)

# Code block 1
# If the cube is not above the table, adjust the lift position and try again
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)