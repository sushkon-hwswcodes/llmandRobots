```python
# Define the target object and get its pose
target_object = "red cube"
position, quaternion, bbox_extent = get_object_pose(target_object)

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion = sample_grasp_pose(target_object)

# Move to a position above the red cube
goto_pose(grasp_position, quaternion, z_approach=0.1)

# Close the gripper to pick up the red cube
close_gripper()

# Move up slightly after picking up the red cube
goto_pose(grasp_position + [0, 0, 0.1], quaternion, z_approach=0.1)
```