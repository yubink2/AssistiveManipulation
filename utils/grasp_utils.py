import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R

from utils.point_cloud_utils import get_point_cloud_from_collision_shapes_specific_link


def rotate_quaternion_by_axis(quaternion, axis='y', degrees=-90):
    rotation = R.from_euler(axis, degrees, degrees=True).as_quat()
    new_quaternion = R.from_quat(quaternion) * R.from_quat(rotation)
    
    return new_quaternion.as_quat()

def filter_transform_matrices_by_position(matrices, x_range, y_range, z_range):
    """
    Filters transformation matrices based on the user-defined range of positions and returns the indices.

    x_range (tuple): The range (min, max) for the x coordinate.
    y_range (tuple): The range (min, max) for the y coordinate.
    z_range (tuple): The range (min, max) for the z coordinate.
    """
    filtered_matrices = []
    indices = []
    for idx, matrix in enumerate(matrices):
        position = matrix[:3, 3]  # Extract the position (x, y, z)
        if (x_range[0] <= position[0] <= x_range[1] and
            y_range[0] <= position[1] <= y_range[1] and
            z_range[0] <= position[2] <= z_range[1]):
            filtered_matrices.append(matrix)
            indices.append(idx)
    return np.array(filtered_matrices), indices

def get_human_arm_pcd_for_grasp(env, scale=None, scale_radius=None, scale_height=None):
    # right_shoulder_pcd = get_point_cloud_from_collision_shapes_specific_link(env.humanoid._humanoid, env.right_shoulder, resolution=30, 
    #                                                                       scale_radius=scale_radius, scale_height=scale_height)
    right_elbow_pcd = get_point_cloud_from_collision_shapes_specific_link(env.humanoid._humanoid, env.right_elbow, resolution=10, 
                                                                          scale_radius=scale_radius, scale_height=scale_height)
    right_wrist_pcd = get_point_cloud_from_collision_shapes_specific_link(env.humanoid._humanoid, env.right_wrist, resolution=10, 
                                                                          scale_radius=scale_radius, scale_height=scale_height)
    pcd = np.vstack((right_elbow_pcd, right_wrist_pcd))
    
    if scale is not None:
        pcd = pcd * scale
    
    # for point in pcd:
    #     env.draw_sphere_marker(position=point, radius=0.01)
    
    dict = {'xyz': pcd}
    return dict

def get_human_arm_pcd_for_grasp_sampler(env, scale=None, scale_radius=None, scale_height=None, client_id=0):
    # right_shoulder_pcd = get_point_cloud_from_collision_shapes_specific_link(env.humanoid._humanoid, env.right_shoulder, resolution=30, 
    #                                                                       scale_radius=scale_radius, scale_height=scale_height)
    right_elbow_pcd = get_point_cloud_from_collision_shapes_specific_link(env.humanoid._humanoid, env.right_elbow, resolution=40, 
                                                                          scale_radius=scale_radius, scale_height=scale_height,
                                                                          skip_hemispherical=True, client_id=client_id)
    # right_wrist_pcd = get_point_cloud_from_collision_shapes_specific_link(env.humanoid._humanoid, env.right_wrist, resolution=20, 
    #                                                                       scale_radius=scale_radius, scale_height=scale_height)
    # pcd = np.vstack((right_elbow_pcd, right_wrist_pcd))

    # if scale is not None:
    #     pcd = pcd * scale
    
    # return pcd

    return right_elbow_pcd

def scale_translation(matrix, scale):
    scaled_matrix = matrix.copy()
    scaled_matrix[:3, 3] *= scale
    return scaled_matrix



def generate_grasps(env):
    # init human arm and get its point cloud
    env.reset_human_arm([2.7, 0.4, -2, 0.5])
    # env.reset_human_arm([2.7, 0.4, -2, 0.8])

    # generate new pcd
    # dict = get_human_arm_pcd_for_grasp(env, scale_radius=0.6, scale_height=1.1)  # scale down the radius
    dict = get_human_arm_pcd_for_grasp(env, scale=0.5)  # scale down the radius
    np.save("pc_human_arm.npy", dict)

    # compute grasps
    global_config = load_config(checkpoint_dir="contact_graspnet_pytorch/checkpoints/contact_graspnet", batch_size=1, arg_configs=[])
    inference(global_config=global_config, 
                ckpt_dir="contact_graspnet_pytorch/checkpoints/contact_graspnet",
                input_paths="pc_human_arm.npy", visualize_results=True)
    
    # grasp results
    grasp_results = np.load("results/predictions_pc_human_arm.npz", allow_pickle=True)
    pred_grasps_cam = grasp_results['pred_grasps_cam'].item()
    scores = grasp_results['scores'].item()
    pred_grasps_cam_values = list(pred_grasps_cam.values())[0]
    scores_values = list(scores.values())[0]
    print(f'original grasps: {len(pred_grasps_cam_values)}')

    ###### DEBUGGING
    top_indices = np.argsort(scores_values)[-20:][::-1]
    top_transformation_matrices = pred_grasps_cam_values[top_indices]
    print(f'original top grasps: {len(top_indices)}')

    for i in range(len(top_transformation_matrices)):
        position = top_transformation_matrices[i][:3,3]
        quaternion = quaternion_from_matrix(top_transformation_matrices[i][:3,:3])
        draw_frame(env, position, quaternion)
    print('')

    top_transformation_matrices = [scale_translation(matrix, scale=2.0) for matrix in top_transformation_matrices]
    for i in range(len(top_transformation_matrices)):
        position = top_transformation_matrices[i][:3,3]
        quaternion = quaternion_from_matrix(top_transformation_matrices[i][:3,:3])
        draw_frame(env, position, quaternion)
    print('')
    sys.exit()
    ######

    # compute x_range, y_range, z_range
    right_wrist_world_pos = np.array(env.bc.getLinkState(env.humanoid._humanoid, env.right_wrist)[0])
    right_elbow_world_pos = np.array(env.bc.getLinkState(env.humanoid._humanoid, env.right_elbow)[0])
    right_elbow_link_pos = np.array(env.bc.getLinkState(env.humanoid._humanoid, env.right_elbow)[4])
    right_elbow_diff = np.abs(right_elbow_world_pos-right_elbow_link_pos)
    sorted_indices = np.argsort(right_elbow_diff)
    smallest_two_indices = sorted_indices[:2]
    other_index = sorted_indices[-1]

    x_range = [right_wrist_world_pos[0]-0.13, right_wrist_world_pos[0]+0.13]
    y_range = [right_wrist_world_pos[1]-0.13, right_wrist_world_pos[1]+0.13]
    z_range = [right_wrist_world_pos[2]-0.13, right_wrist_world_pos[2]+0.13]

    # filter matrices based on the ranges
    filtered_matrices, filtered_indices = filter_transform_matrices_by_position(pred_grasps_cam_values, x_range, y_range, z_range)
    filtered_scores = scores_values[filtered_indices]
    print(f'filtered grasps: {len(filtered_matrices)}')

    if len(filtered_matrices) == 0:
        raise ValueError('no grasp available')

    # sort by highest score
    filtered_idx_score = list(zip(filtered_indices, filtered_scores))
    filtered_idx_score_sorted = sorted(filtered_idx_score, key=lambda x: x[1], reverse=True)
    filtered_idx_sorted = [pair[0] for pair in filtered_idx_score_sorted][:30]
    top_transformation_matrices = pred_grasps_cam_values[filtered_idx_sorted]
    print(f'top filtered grasps: {len(top_transformation_matrices)}')

    # human arm parameters for axis rotation
    world_to_right_elbow = env.bc.getLinkState(env.humanoid._humanoid, env.right_elbow)[:2]
    right_elbow_to_world = env.bc.invertTransform(world_to_right_elbow[0], world_to_right_elbow[1])
    
    grasp_samples = []
    for i in range(len(top_transformation_matrices)):
        # rotate frames and append more grasps
        degrees_list = [0, 90, 180, 270]
        for degrees in degrees_list:
            position = top_transformation_matrices[i][:3,3]
            quaternion = quaternion_from_matrix(top_transformation_matrices[i][:3,:3])

            # rotate matrices by 'y' axis of human arm
            right_elbow_to_grasp = env.bc.multiplyTransforms(right_elbow_to_world[0], right_elbow_to_world[1],
                                                            position, quaternion)
            world_to_right_elbow_rotated = [world_to_right_elbow[0], rotate_quaternion_by_axis(world_to_right_elbow[1], axis='y', degrees=degrees)]
            world_to_grasp_rotated = env.bc.multiplyTransforms(world_to_right_elbow_rotated[0], world_to_right_elbow_rotated[1],
                                                                right_elbow_to_grasp[0], right_elbow_to_grasp[1])

            # rotate matrices such that panda eef --> ur5 eef
            position = world_to_grasp_rotated[0]
            quaternion = world_to_grasp_rotated[1]
            quaternion = rotate_quaternion_by_axis(quaternion, axis='y', degrees=-90)
            # quaternion = rotate_quaternion_by_axis(quaternion, axis='x', degrees=180)

            # if other_index != 2:
            #     quaternion = rotate_quaternion_by_axis(quaternion, axis='x', degrees=180)
            # else:
            #     quaternion = rotate_quaternion_by_axis(quaternion, axis='x', degrees=90)

            if degrees == 90:
                draw_frame(env, position=position, quaternion=quaternion)

            grasp_samples.append([list(position), list(quaternion)])

    print(f'recomputed grasps: {len(grasp_samples)}')

    # test each grasp sample
    q_R_grasp_samples = []
    for grasp in grasp_samples:
        world_to_eef = env.bc.multiplyTransforms(grasp[0], grasp[1],
                                                    env.eef_grasp_to_eef[0], env.eef_grasp_to_eef[1])
        q_R_grasp = env.bc.calculateInverseKinematics(env.robot.id, env.robot.eef_id, 
                                                    world_to_eef[0], world_to_eef[1],
                                                    env.robot.arm_lower_limits, env.robot.arm_upper_limits, env.robot.arm_joint_ranges, env.robot.arm_rest_poses,
                                                    maxNumIterations=50)
        q_R_grasp = [q_R_grasp[i] for i in range(len(env.robot.arm_controllable_joints))]

        env.reset_robot(env.robot, q_R_grasp)
        eef_pose = env.bc.getLinkState(env.robot.id, env.robot.eef_id)[:2]
        dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))

        if not env.collision_fn_2(q_R_grasp) and dist <= 0.05:
            q_R_grasp_samples.append(q_R_grasp)

    print(f'no collision grasps: {len(q_R_grasp_samples)}')
    
    if len(q_R_grasp_samples) == 0:
        raise ValueError('no grasp available')
    
    # # with the highest score
    # env.reset_robot(env.robot, q_R_grasp_samples[0])
    for q_R in q_R_grasp_samples:
        env.reset_robot(env.robot, q_R)
    print('here')

# def convert_opencv_to_pybullet(opencv_transform):
#     """
#     Convert a transformation matrix from OpenCV coordinates to PyBullet coordinates.
    
#     Args:
#     opencv_transform (np.ndarray): A 4x4 transformation matrix in OpenCV coordinates.
    
#     Returns:
#     np.ndarray: A 4x4 transformation matrix in PyBullet coordinates.
#     """
#     # Create a matrix to convert from OpenCV to PyBullet coordinate system
#     opencv_to_pybullet = np.array([
#         [0, 0, 1, 0],
#         [-1, 0, 0, 0],
#         [0, -1, 0, 0],
#         [0, 0, 0, 1]
#     ])
    
#     # Convert the transformation matrix
#     pybullet_transform = opencv_to_pybullet @ opencv_transform @ np.linalg.inv(opencv_to_pybullet)
    
#     return pybullet_transform

# def convert_opengl_to_pybullet(opengl_transform):
#     """
#     Convert a transformation matrix from OpenGL coordinates to PyBullet coordinates.
    
#     Args:
#     opencv_transform (np.ndarray): A 4x4 transformation matrix in OpenGL coordinates.
    
#     Returns:
#     np.ndarray: A 4x4 transformation matrix in PyBullet coordinates.
#     """
#     # Create a matrix to convert from OpenCV to PyBullet coordinate system
#     opengl_to_pybullet = np.array([
#         [0, 0, -1, 0],
#         [-1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 0, 1]
#     ])
    
#     # Convert the transformation matrix
#     pybullet_transform = opengl_to_pybullet @ opengl_transform @ np.linalg.inv(opengl_to_pybullet)
    
#     return pybullet_transform
