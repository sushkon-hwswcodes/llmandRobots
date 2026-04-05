```python
# Get the pose of the red cube
cube_position, _, _ = get_object_pose("red_cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red_cube")

# Move the robot to the grasp position with an approach motion
goto_pose(grasp_position, grasp_quaternion_wxyz, z_approach=0.1)

# Close the gripper to pick up the cube
close_gripper()

# Lift the red cube by moving the hand parallel to the XY plane
lift_position = cube_position + np.array([0, 0, 0.2])
goto_pose(lift_position, grasp_quaternion_wxyz)

# Open the gripper fully
open_gripper()
```