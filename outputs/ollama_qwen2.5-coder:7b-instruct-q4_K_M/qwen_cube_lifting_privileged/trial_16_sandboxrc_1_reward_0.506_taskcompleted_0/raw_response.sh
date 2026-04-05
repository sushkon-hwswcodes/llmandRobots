```python
# Get the pose of the red cube
cube_position, cube_quaternion_wxyz, _ = get_object_pose("red cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Move to a safe approach position above the cube
goto_pose(grasp_position, grasp_quaternion_wxyz, z_approach=0.1)

# Close the gripper to pick up the red cube
close_gripper()

# Lift the red cube by moving vertically upwards
lift_position = cube_position + np.array([0, 0, 0.2])
goto_pose(lift_position, grasp_quaternion_wxyz)
```