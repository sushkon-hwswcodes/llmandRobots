# Code block 0
import numpy as np

# Get the position and quaternion of the red cube
cube_pos, cube_quat, _ = get_object_pose("red_cube", return_bbox_extent=False)

# Approach pose above the cube
approach_pos = cube_pos.copy()
approach_pos[2] += 0.1
goto_pose(approach_pos, cube_quat, z_approach=0.1)

# Grasp pose slightly below the cube
grasp_pos = cube_pos.copy()
grasp_pos[2] += 0.05 * get_object_pose("cube").bbox_extent[2]
goto_pose(grasp_pos, cube_quat)

# Close the gripper to grasp the cube
close_gripper()

# Lift the cube above the table
lift_pos = cube_pos.copy()
lift_pos[2] = 0.4  # Adjust height as needed
goto_pose(lift_pos, cube_quat)