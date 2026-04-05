```python
import numpy as np

# Get the pose of the red cube
cube_position, _, _ = get_object_pose("red cube")

# Go to a position slightly above the red cube
goto_pose(cube_position + np.array([0, 0, 0.1]), np.array([0, 0, 1, 0]))

# Lift over the cube
close_gripper()
goto_pose(cube_position + np.array([0, 0, 0.5]), np.array([0, 0, 1, 0]))
```