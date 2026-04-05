# Code block 0
import numpy as np

# Get the pose of the red cube
red_cube_pose = get_object_pose("red cube", return_bbox_extent=True)
red_cube_position, _, _ = red_cube_pose

# Lift the cube by moving straight upwards
lifted_pose = (red_cube_position[0], red_cube_position[1], red_cube_position[2] + 0.2)  # Move up by 20 cm
goto_pose(lifted_pose, (0, 0, 1, 0))

# Open the gripper to grab the cube
close_gripper()

# Lift the cube further while holding it
lifted_pose = (red_cube_position[0], red_cube_position[1], red_cube_position[2] + 0.3)  # Move up by another 30 cm
goto_pose(lifted_pose, (0, 0, 1, 0))