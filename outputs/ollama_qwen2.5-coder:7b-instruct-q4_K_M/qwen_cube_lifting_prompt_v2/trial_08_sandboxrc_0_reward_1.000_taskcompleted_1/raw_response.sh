```python
import numpy as np

# Get the pose of the red cube
cube_pose, cube_quat, _ = get_object_pose("red_cube")

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Move to the approach position and then to the grasp position
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to pick up the red cube
close_gripper()

# Lift the cube by moving it above the table
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)
```