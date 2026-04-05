# Code block 0
# Get the pose of the red cube
red_cube_pose, _, _ = get_object_pose("red cube")
red_cube_position, red_cube_quaternion_wxyz = red_cube_pose[:3], red_cube_pose[3:7]

# Sample a grasp pose for the red cube
grasp_position, grasp_quaternion_wxyz = sample_grasp_pose("red cube")

# Move to the approach position above the object
goto_pose(grasp_position, grasp_quaternion_wxyz, z_approach=0.1)

# Close the gripper to pick up the red cube
close_gripper()

# Go to a height where it's stable and lifted slightly
lifted_pose = [red_cube_position[0], red_cube_position[1], red_cube_position[2] + 0.1]
goto_pose(lifted_pose, grasp_quaternion_wxyz)

# Open the gripper after picking up the red cube
open_gripper()

print("Red cube picked up and lifted successfully.")