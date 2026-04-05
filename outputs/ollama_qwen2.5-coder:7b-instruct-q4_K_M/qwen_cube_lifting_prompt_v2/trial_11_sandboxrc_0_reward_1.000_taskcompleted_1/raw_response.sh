```python
import numpy as np

# Get the poses for the red cube and sample a grasp pose for it
cube_pose, _, _ = get_object_pose("red_cube")
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Approach and then grasp the cube
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to grab the cube
close_gripper()

# Lift the cube above the table
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1
goto_pose(lift_pos, grasp_quat)
```