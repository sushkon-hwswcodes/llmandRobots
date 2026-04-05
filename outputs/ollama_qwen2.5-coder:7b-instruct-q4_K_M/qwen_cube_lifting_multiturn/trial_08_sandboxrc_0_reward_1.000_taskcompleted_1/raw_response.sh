import numpy as np

# Move to a position near the red cube
red_cube_pose = get_object_pose("red cube")
position = red_cube_pose[0] - np.array([0, 0.1, 0])  # Adjust the offset based on robot reach and cube proximity
quaternion_wxyz = (0, 0, 0, 1)  # Using a known orientation for simplicity

goto_pose(position, quaternion_wxyz)

# Open the gripper to prepare for picking up the red cube
open_gripper()

# Move closer to pick up the red cube
z_approach = -0.05
goto_pose(red_cube_pose[0], red_cube_pose[1], z_approach)

# Close the gripper to grab the red cube
close_gripper()

# Lift the red cube slightly above its original position
lift_height = 0.2
new_position = red_cube_pose[0] + np.array([0, 0, lift_height])
goto_pose(new_position, red_cube_pose[1])