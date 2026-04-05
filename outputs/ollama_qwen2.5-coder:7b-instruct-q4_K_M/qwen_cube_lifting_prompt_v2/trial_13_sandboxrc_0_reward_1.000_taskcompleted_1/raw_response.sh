```python
import numpy as np

# Get the pose of the red cube
cube_pos, cube_quat, bbox_extent = get_object_pose("red_cube")

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Approach and then grasp the cube
goto_pose(grasp_pos, grasp_quat)
close_gripper()

# Lift the cube above the table
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)
```