# ur5, pybullet
import os, inspect
import os.path as osp
import pybullet as p
import math
import sys

import pybullet_data
from pybullet_ur5.robot import UR5Robotiq85, Panda
from pybullet_utils.bullet_client import BulletClient
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.debug_utils import *
from utils.transform_utils import *

# humanoid
from deep_mimic.env.motion_capture_data import MotionCaptureData
from deep_mimic.mocap.humanoid_with_rev_xyz import Humanoid
from deep_mimic.mocap.humanoid_with_rev_xyz import HumanoidPose

# mppi planner (ramp)
from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower
from mppi_planning.mppi_human_clamping import MPPI_H_Clamp

# point cloud
import open3d as o3d
from utils.point_cloud_utils import *

# grasp generation
from utils.grasp_utils import *
from grasp_sampler.antipodal_grasp_sampler import *

# environments
from manipulation_demo import ManipulationDemo
from wiping_task.wiping_planner import WipingDemo

# utils
from utils.collision_utils import get_collision_fn

# video recording
import cv2

# urdf paths
robot_urdf_location = 'pybullet_ur5/urdf/ur5_robotiq_85.urdf'
scene_urdf_location = 'resources/environment/environment.urdf'
control_points_location = 'resources/ur5_control_points/T_control_points.json'
control_points_number = 55

# UR5 parameters
LINK_FIXED = 'base_link'
LINK_EE = 'ee_link'
LINK_SKELETON = [
    'shoulder_link',
    'upper_arm_link',
    'forearm_link',
    'wrist_1_link',
    'wrist_2_link',
    'wrist_3_link',
    'ee_link',
]

# Frame capture helper functions
def capture_frame(bc, frame_dir, frame_count, width=640, height=480):
    """Capture a frame from the PyBullet simulation."""
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0],
                                                      distance=2,
                                                      yaw=0,
                                                      pitch=-30,
                                                      roll=0,
                                                      upAxisIndex=2,
                                                      physicsClientId=bc._client)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height,
                                               nearVal=0.1, farVal=100.0, physicsClientId=bc._client)

    (_, _, px, _, _) = p.getCameraImage(width=width, height=height,
                                        viewMatrix=view_matrix,
                                        projectionMatrix=proj_matrix,
                                        physicsClientId=bc._client)
    img = np.reshape(px, (height, width, 4))  # RGBA
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(f"{frame_dir}/frame_{frame_count:04d}.png", img)


def save_video_from_frames(frame_dir, output_file, fps=20):
    """Convert captured frames into a video."""
    img_array = []
    for filename in sorted(os.listdir(frame_dir)):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(frame_dir, filename))
            img_array.append(img)

    height, width, _ = img_array[0].shape
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img in img_array:
        out.write(img)
    out.release()

def wiping_loop(wiping_env, manip_env, q_H, total_targets_cleared, q_robot, q_robot_2_init):
    # initialize environments
    current_joint_angles = q_robot_2_init
    manip_env.targets_util.update_targets()
    wiping_env.lock_robot_arm_joints(wiping_env.robot, q_robot)

    arms = ['upperarm', 'forearm']
    targets_cleared = 0
    for arm in arms:
        # compute wiping trajectory
        feasible_targets_found = wiping_env.reset_wiping_setup(q_H, arm)
        if not feasible_targets_found:
            continue
        
        robot_traj = wiping_env.compute_feasible_targets_robot_traj()
        if len(robot_traj) == 0:
            print(f'{arm}: no valid wiping trajectory!')
            continue

        robot_traj = wiping_env.interpolate_trajectory(robot_traj, alpha=0.5)
        robot_traj = wiping_env.interpolate_trajectory(robot_traj, alpha=0.25)

        # compute feasible targets parameters
        feasible_targets_pos_world, feasible_targets_orn_world, feasible_targets_count, feasible_targets_indices, init_q_R, arm_side = wiping_env.targets_util.get_feasible_targets_lists()
        feasible_targets = manip_env.targets_util.get_feasible_targets_given_indices(feasible_targets_indices, arm_side)
        manip_env.targets_util.set_feasible_targets_lists(feasible_targets_pos_world, feasible_targets_orn_world, feasible_targets, feasible_targets_count, feasible_targets_indices, init_q_R, arm_side)
        manip_env.targets_util.mark_feasible_targets()

        # move robot_2 to wiping initial config
        manip_env.reset_robot(manip_env.robot_2, manip_env.targets_util.init_q_R)   ####
        manip_env.attach_tool()
        # move_robot_loop(manip_env, robot=manip_env.robot_2, other_robot=manip_env.robot, 
        #                 q_robot_init=current_joint_angles, q_robot_goal=manip_env.targets_util.init_q_R, q_other_robot=q_robot, q_H=q_H)

        # execute wiping trajectory
        for q_R in robot_traj:
            for _ in range(50):
                if manip_env.human_cid is None:
                    manip_env.reset_human_arm(q_H)
                manip_env.move_robot(manip_env.robot_2, q_R)
                manip_env.move_robot(manip_env.robot, q_robot)
                manip_env.bc.stepSimulation()
            # time.sleep(0.3)
            new_target, indices_to_delete = manip_env.targets_util.get_new_contact_points(targeted_arm=arm)
            manip_env.targets_util.remove_contacted_feasible_targets(indices_to_delete, arm)
            wiping_env.targets_util.remove_contacted_feasible_targets(indices_to_delete, arm)

            targets_cleared += new_target
            total_targets_cleared += new_target

        manip_env.targets_util.remove_targets()
        manip_env.targets_util.unmark_feasible_targets()
        manip_env.targets_util.update_targets()
        manip_env.detach_tool()

        wiping_env.targets_util.remove_targets()
        wiping_env.targets_util.update_targets()

        current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_2)

    # move robot_2 back to rest poses
    manip_env.reset_robot(manip_env.robot_2, q_robot_2_init)   ####
    # move_robot_loop(manip_env, robot=manip_env.robot_2, other_robot=manip_env.robot, 
    #                 q_robot_init=current_joint_angles, q_robot_goal=q_robot_2_init, q_other_robot=q_robot, q_H=q_H)
    wiping_env.reset_robot(wiping_env.robot_2, q_robot_2_init)

    return targets_cleared, total_targets_cleared

def arm_manipulation_loop(manip_env, q_robot_2, q_robot_init, q_robot_goal, q_H_init, world_to_eef_goal):
    # Step 0: instantiate a new motion planning problem
    trajectory_planner = manip_env.init_traj_planner(manip_env.world_to_robot_base, clamp_by_human=True)
    trajectory_follower = manip_env.init_traj_follower(manip_env.world_to_robot_base)
    
    # Step 1: move robot to grasping pose
    manip_env.reset_robot(manip_env.robot_2, q_robot_2)
    manip_env.reset_robot(manip_env.robot, q_robot_init)
    manip_env.reset_human_arm(q_H_init)

    # Step 2: attach human arm to eef
    env_pcd, right_arm_pcd, _ = manip_env.compute_env_pcd(manip_env.robot_2)
    T_eef_to_object, T_object_to_world = manip_env.attach_human_arm_to_eef(right_arm_pcd, attach_to_gripper=True, trajectory_planner=trajectory_planner)

    # Step 3: trajectory after grasping
    trajectory_planner = manip_env.init_mppi_planner(trajectory_planner, q_robot_init, q_robot_goal, clamp_by_human=True)
    traj = manip_env.get_mppi_trajectory(trajectory_planner, q_robot_init)
    previous_update_time = time.time()
    update_second = 20  # sec

    # Step 4: initialize trajectory planner & follower
    trajectory_planner.update_obstacle_pcd(env_pcd)
    trajectory_follower.update_obstacle_pcd(env_pcd)
    trajectory_follower.update_trajectory(traj)
    trajectory_follower.attach_to_gripper(object_type="pcd", object_geometry=right_arm_pcd,
                                            T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world,
                                            T_world_to_human_base=manip_env.T_world_to_human_base, T_right_elbow_joint_to_cp=manip_env.T_right_elbow_joint_to_cp,
                                            human_arm_lower_limits=manip_env.human_arm_lower_limits, human_arm_upper_limits=manip_env.human_arm_upper_limits)
    
    current_joint_angles = q_robot_init
    current_human_joint_angles = q_H_init
    world_to_eef = manip_env.bc.getLinkState(manip_env.robot.id, manip_env.robot.eef_id)[:2]

    # #### DEBUGGINGGG
    # for q in traj:
    #     for _ in range (100):
    #         manip_env.move_robot(manip_env.robot_2, q_robot_2)
    #         manip_env.move_robot(manip_env.robot, q)
    #         manip_env.bc.stepSimulation()
    #     time.sleep(0.3)

    # Step 5: simulation loop
    while True:
        # if near goal, execute rest of trajectory and end simulation loop
        if manip_env.is_near_goal_W_space(world_to_eef, world_to_eef_goal):
            traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles)
            traj = manip_env.interpolate_trajectory(traj, alpha=0.5)
            for q in traj:
                for _ in range (100):
                    manip_env.move_robot(manip_env.robot_2, q_robot_2)
                    manip_env.move_robot(manip_env.robot, q)
                    manip_env.bc.stepSimulation()
                manip_env.targets_util.update_targets()
                # time.sleep(0.3)
            break

        # get velocity command
        prev_time = time.time()
        velocity_command = trajectory_follower.follow_trajectory(current_joint_angles, current_human_joint_angles)[0]
        current_time = time.time()
        # print('following time: ', current_time-prev_time)

        # update trajectory 
        if current_time-previous_update_time > update_second:
            print('replanning...')
            traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles)
            trajectory_follower.update_trajectory(traj)
            previous_update_time = time.time()

        # if valid velocity command, move robot
        else:
            joint_angles = current_joint_angles + velocity_command * 0.2
            for _ in range (100):
                manip_env.move_robot(manip_env.robot_2, q_robot_2)
                manip_env.move_robot(manip_env.robot, joint_angles)
                manip_env.bc.stepSimulation()
            manip_env.targets_util.update_targets()

            # save current_joint_angle
            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
            current_human_joint_angles = manip_env.get_human_joint_angles()
            world_to_eef = manip_env.bc.getLinkState(manip_env.robot.id, manip_env.robot.eef_id)[:2]

    print('arm manipulation loop is done')

def move_robot_loop(manip_env, robot, other_robot, q_robot_init, q_robot_goal, q_other_robot, q_H):
    # Step 0: instantiate a new motion planning problem
    if robot == manip_env.robot:
        world_to_robot_base = manip_env.world_to_robot_base
    elif robot == manip_env.robot_2:
        world_to_robot_base = manip_env.world_to_robot_2_base
    else:
        raise ValueError('invalid robot!')
    
    trajectory_planner = manip_env.init_traj_planner(world_to_robot_base, clamp_by_human=False)
    trajectory_follower = manip_env.init_traj_follower(world_to_robot_base)
    
    # Step 1: initialize trajectory planner & follower
    if manip_env.human_cid is None:
        manip_env.reset_human_arm(q_H)
    trajectory_planner = manip_env.init_mppi_planner(trajectory_planner, q_robot_init, q_robot_goal, clamp_by_human=False)
    env_pcd, right_arm_pcd, right_shoulder_pcd = manip_env.compute_env_pcd(other_robot)
    env_pcd = np.vstack((env_pcd, right_arm_pcd, right_shoulder_pcd))
    trajectory_planner.update_obstacle_pcd(env_pcd)
    trajectory_follower.update_obstacle_pcd(env_pcd)
    current_joint_angles = q_robot_init

    # TODO attach tool to planner and follower....................................................
    # if robot == manip_env.robot_2: 

    # Step 2: compute trajectory
    traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles=q_robot_init)
    trajectory_follower.update_trajectory(traj)
    previous_update_time = time.time()
    update_second = 3  # sec

    # Step 3: simulation loop
    while True:
        # if near goal, execute rest of trajectory and terminate the simulation loop
        if manip_env.is_near_goal_C_space(current_joint_angles, q_robot_goal):
            traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles)
            for q in traj:
                for _ in range (100):
                    if manip_env.human_cid is None:
                        manip_env.reset_human_arm(q_H)
                    manip_env.move_robot(robot, q)
                    manip_env.move_robot(other_robot, q_other_robot)
                    manip_env.bc.stepSimulation()
                time.sleep(0.3)
            break

        # get velocity command
        prev_time = time.time()
        velocity_command = trajectory_follower.follow_trajectory(current_joint_angles, current_human_joint_angles=[])[0]
        current_time = time.time()
        # print('following time: ', current_time-prev_time)

        # update trajectory 
        if current_time-previous_update_time > update_second:
            print('replanning...')
            traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles)
            trajectory_follower.update_trajectory(traj)
            previous_update_time = time.time()

        # move robot
        else:
            joint_angles = current_joint_angles + velocity_command * 0.2
            for _ in range (100):
                if manip_env.human_cid is None:
                    manip_env.reset_human_arm(q_H)
                manip_env.move_robot(robot, joint_angles)
                manip_env.move_robot(other_robot, q_other_robot)
                manip_env.bc.stepSimulation()

            # save current_joint_angle
            current_joint_angles = manip_env.get_robot_joint_angles(robot)

    print('move robot loop is done')

if __name__ == '__main__':
    # Initialize environments
    wiping_env = WipingDemo()
    manip_env = ManipulationDemo()
    wiping_env.reset()
    manip_env.reset()

    # Create directory for frames
    frame_dir = "frames/"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    # Video frame rate
    fps = 20
    frame_count = 0

    # Initial joint states
    q_robot_init = manip_env.robot.arm_rest_poses
    q_robot_2_init = manip_env.robot_2.arm_rest_poses
    q_H_init = manip_env.human_rest_poses

    # First wiping iteration with rest poses
    start_time = time.time()
    success_rate = 0.0
    total_targets_cleared = 0
    total_targets = wiping_env.targets_util.total_target_count

    manip_env.reset_human_arm(q_H_init)
    targets_cleared, total_targets_cleared = wiping_loop(wiping_env, manip_env, q_H_init, total_targets_cleared,
                                                         q_robot_init, q_robot_2_init)
    success_rate = total_targets_cleared / total_targets
    print(f'total_targets_cleared: {total_targets_cleared}/{total_targets}')

    # Grasp generation
    q_H_init = manip_env.human_rest_poses
    q_R_grasp_samples, grasp_pose_samples, best_q_R_grasp, best_world_to_grasp = manip_env.generate_grasps(q_H_init)
    manip_env.compute_grasp_parameters(q_H_init, best_q_R_grasp, best_world_to_grasp)

    # Simulation loop until threshold is met...
    current_joint_angles = q_robot_init
    target_joint_angles = best_q_R_grasp
    current_robot_2_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_2)
    current_human_joint_angles = q_H_init
    for _ in range(2):
        # Capture frame during the simulation
        capture_frame(manip_env.bc, frame_dir, frame_count)
        frame_count += 1

        # Find q_H_goal and q_R_goal using the grasp
        valid_grasp, feasible_targets_found = False, False
        for _ in range(10000):
            q_H_goal, world_to_right_elbow = wiping_env.get_valid_q_H()
            feasible_targets_found_on_upperarm = wiping_env.reset_wiping_setup(q_H_goal, targeted_arm='upperarm')
            feasible_targets_found_on_forearm = wiping_env.reset_wiping_setup(q_H_goal, targeted_arm='forearm')
            feasible_targets_found = feasible_targets_found_on_upperarm or feasible_targets_found_on_forearm
            if not feasible_targets_found:
                continue
            valid_grasp, q_R_goal, world_to_eef_goal = manip_env.compute_q_R_goal(world_to_right_elbow)
            if valid_grasp:
                break
        if not valid_grasp:
            raise ValueError('valid q_H not found for the grasp!')

        # Arm manipulation
        if manip_env.human_cid is None:
            # Move to grasp pose for the first arm manipulation
            manip_env.reset_human_arm(current_human_joint_angles)
            move_robot_loop(manip_env, robot=manip_env.robot, other_robot=manip_env.robot_2,
                            q_robot_init=current_joint_angles, q_robot_goal=target_joint_angles,
                            q_other_robot=current_robot_2_joint_angles, q_H=current_human_joint_angles)
            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)

        arm_manipulation_loop(manip_env, q_robot_2=current_robot_2_joint_angles,
                              q_robot_init=current_joint_angles, q_robot_goal=q_R_goal, q_H_init=current_human_joint_angles,
                              world_to_eef_goal=world_to_eef_goal)
        current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
        current_human_joint_angles = manip_env.get_human_joint_angles()

        # N-th wiping iteration
        targets_cleared, total_targets_cleared = wiping_loop(wiping_env, manip_env, current_human_joint_angles, total_targets_cleared,
                                                             q_robot=current_joint_angles, q_robot_2_init=current_robot_2_joint_angles)

        # Check if wiping threshold is reached
        success_rate = total_targets_cleared / total_targets
        print(f'success_rate: {success_rate}, total_targets_cleared: {total_targets_cleared}/{total_targets}')
        if success_rate >= 0.8:
            break

        # Save states
        current_human_joint_angles = manip_env.get_human_joint_angles()
        current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
        current_robot_2_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_2)

    total_time = time.time() - start_time
    print(f'success_rate: {success_rate}, total_targets_cleared: {total_targets_cleared}/{total_targets}')
    print(f'total simulation time: {total_time}')
    print('done')

    # Save the video after simulation is complete
    output_file = "simulation_video.mp4"
    save_video_from_frames(frame_dir, output_file, fps=fps)
    print(f"Video saved as {output_file} with {fps} FPS")