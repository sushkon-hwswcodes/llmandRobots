```python
import numpy as np

# Sample a grasp pose for the red cube
grasp_pos, grasp_quat = sample_grasp_pose("red_cube")

# Open the gripper to prepare for picking up the cube
open_gripper()

# Go to the approach position above the cube
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)

# Move to the exact grasp pose
goto_pose(grasp_pos, grasp_quat)

# Close the gripper to pick up the cube
close_gripper()

# Define the lift position above the table
lift_pos = grasp_pos.copy()
lift_pos[2] += 0.1

# Go to the lift position
goto_pose(lift_pos, grasp_quat)
```