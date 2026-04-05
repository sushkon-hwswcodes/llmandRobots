```python
# Get the pose of the red cube
red_cube_pose = get_object_pose("red cube", return_bbox_extent=False)

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Move to the z-approach position of the grasp pose
goto_pose(grasp_position, grasp_quaternion_wxyz, z_approach=0.1)

# Close the gripper to pick up the red cube
close_gripper()

# Lift the red cube by moving to a higher position along the Z-axis
lift_height = 0.3  # Example lift height in meters
lift_position = grasp_position.copy()
lift_position[2] += lift_height

goto_pose(lift_position, grasp_quaternion_wxyz)

# Open the gripper to release the red cube
open_gripper()
```