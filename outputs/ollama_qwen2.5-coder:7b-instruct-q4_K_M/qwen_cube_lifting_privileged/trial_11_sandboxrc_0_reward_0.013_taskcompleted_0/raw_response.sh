```python
import numpy as np

# Get the pose of the red cube
cube_position, _, _ = get_object_pose("red_cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red_cube")

# Move to the grasp position and pre-grasp z_height above the object
z_approach = 0.1
goto_pose(grasp_position + np.array([0, 0, z_approach]), grasp_quaternion_wxyz)

# Close the gripper to pick up the cube
close_gripper()

# Lift the cube vertically by a small height
lift_offset = 0.2
goto_pose(grasp_position + np.array([0, 0, lift_offset]), grasp_quaternion_wxyz)

# Open the gripper to release the cube
open_gripper()
```