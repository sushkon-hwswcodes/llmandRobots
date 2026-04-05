```python
# Get the pose of the red cube
cube_position, quaternion_wxyz, bbox_extent = get_object_pose("red cube", return_bbox_extent=True)

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Go to the position of the red cube with an approach distance
goto_pose(cube_position + [0, 0, 0.1], quaternion_wxyz, z_approach=0.1)

# Open the gripper
open_gripper()

# Approach and pick up the red cube slightly above its position
goto_pose(grasp_position, grasp_quaternion_wxyz, z_approach=0.2)

# Close the gripper to grasp the red cube
close_gripper()

# Lift the red cube
goto_pose(gray_position, grasp_quaternion_wxyz, z_approach=0.3)
```