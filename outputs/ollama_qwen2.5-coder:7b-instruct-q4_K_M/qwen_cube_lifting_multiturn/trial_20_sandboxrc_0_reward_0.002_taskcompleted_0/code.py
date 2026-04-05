# Code block 0
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

# Code block 1
import numpy as np

# Get the current position of the gripper
current_gripper_position, _, _ = get_object_pose("gripper")

# Move the robot slightly up from the current position before lifting the cube again for stability
goto_pose(current_gripper_position + np.array([0, 0, 0.1]), (0, 0, 1, 0), z_approach=0)

# Lift the red cube above the initial hand position to avoid collision with ground or other objects
lift_height = 0.3
new_lift_position = current_gripper_position + np.array([0, 0, lift_height])
goto_pose(new_lift_position, (0, 0, 1, 0))

# Open the gripper to release the cube
open_gripper()

# Code block 2
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

# Open the gripper to release the cube
open_gripper()