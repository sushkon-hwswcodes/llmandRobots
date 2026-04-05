```python
# Get the pose of the red cube
cube_position, _, _ = get_object_pose("red cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion = sample_grasp_pose("red cube")

# Move the robot to the grip position above the cube
goto_pose(grasp_position + np.array([0, 0, 0.1]), grasp_quaternion, z_approach=0.2)

# Close the gripper to pick up the red cube
close_gripper()

# Lifting height
lift_height = 0.3

# Calculate the new position to lift the cube
new_lift_position = cube_position + np.array([0, 0, lift_height])

# Move the robot to lift the red cube
goto_pose(new_lift_position, grasp_quaternion)
```