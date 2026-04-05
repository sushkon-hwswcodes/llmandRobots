# Code block 0
import numpy as np

def get_red_cube_pose():
    position, _, _ = get_object_pose("red cube")
    return position

def sample_grasp_pose_for_cube():
    position, quaternion_wxyz = sample_grasp_pose("red cube")
    return position, quaternion_wxyz

def goto_and_grab(position, quaternion_wxyz):
    z_approach = 0.1
    goto_pose(position, quaternion_wxyz, z_approach)
    close_gripper()

def main():
    # Get the pose of the red cube
    red_cube_position = get_red_cube_pose()
    
    # Sample a grasp pose for the red cube
    grasp_position, grasp_quaternion = sample_grasp_pose_for_cube()
    
    # Move to the grasp position and grab the red cube
    goto_and_grab(grasp_position, grasp_quaternion)
    
    # Raise the cube (simple lift up along z-axis)
    final_z_approach = 0.2
    goto_pose(red_cube_position, [0, 0, 1, 0], final_z_approach)

main()