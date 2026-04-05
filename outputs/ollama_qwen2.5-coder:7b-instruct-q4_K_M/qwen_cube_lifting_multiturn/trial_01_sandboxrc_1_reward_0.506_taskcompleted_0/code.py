# Code block 0
# Get the pose of the red cube
red_cube_position, _, _ = get_object_pose("red cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Move the robot to the grasp position
goto_pose(grasp_position, grasp_quaternion_wxyz)

# Close the gripper to pick up the red cube
close_gripper()

# Lift the red cube by moving slightly along the Z-axis
lift_height = 0.1
goto_pose(grasp_position + np.array([0, 0, lift_height]), grasp_quaternion_wxyz)

# Code block 1
import numpy as np

# Get the pose of the red cube
red_cube_position, _, _ = get_object_pose("red cube")

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Move the robot to the grasp position
goto_pose(grasp_position, grasp_quaternion_wxyz)

# Close the gripper to pick up the red cube
close_gripper()

# Lift the red cube by moving slightly along the Z-axis
lift_height = 0.1
goto_pose(grasp_position + np.array([0, 0, lift_height]), grasp_quaternion_wxyz)