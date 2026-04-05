```python
import numpy as np

# Get the position and quaternion of the red cube
cube_position, cube_quaternion, _ = get_object_pose("red_cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion = sample_grasp_pose("red_cube")

# Go to the grasp position with an approach distance of 0.1 meters
goto_pose(grasp_position, grasp_quaternion, z_approach=0.1)

# Close the gripper to pick up the red cube
close_gripper()

# Sample a new pose to lift the red cube higher
lift_position = np.array([cube_position[0], cube_position[1], 0.2])
goto_pose(lift_position, grasp_quaternion)
```