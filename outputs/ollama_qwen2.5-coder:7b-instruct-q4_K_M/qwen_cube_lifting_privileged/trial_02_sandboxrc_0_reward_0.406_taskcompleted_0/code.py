# Code block 0
# Get the pose of the red cube
cube_position, cube_quaternion_wxyz, _ = get_object_pose("red cube")

# Sample a grasp pose for the cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Approach to the cube using the sampled grasp position
goto_pose(grasp_position, grasp_quaternion_wxyz, z_approach=0.1)

# Close the gripper to pick up the cube
close_gripper()

# Lift the cube by moving in the Z direction
lift_amount = 0.2
new_z_position = cube_position[2] + lift_amount
goto_pose(cube_position[:3], grasp_quaternion_wxyz, z_approach=0.0)

# Open the gripper after lifting the cube
open_gripper()