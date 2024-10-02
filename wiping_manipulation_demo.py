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
from grasp_generation_demo import GraspDemo

# utils
from utils.collision_utils import get_collision_fn


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

def wiping_loop(wiping_env, manip_env, q_H, total_targets_cleared, q_robot, q_robot_2_init):
    # initialize environments
    current_joint_angles = q_robot_2_init
    manip_env.lock_human_joints(q_H)  ####
    manip_env.targets_util.update_targets()
    wiping_env.lock_robot_arm_joints(wiping_env.robot, q_robot)

    arms = ['upperarm', 'forearm']
    targets_cleared = 0
    for arm in arms:
        for _ in range(5):
            # compute feasible targets & wiping trajectory
            feasible_targets_found = wiping_env.reset_wiping_setup(q_H, arm)
            if not feasible_targets_found:
                print(f'{arm} feasible targets not found!')
                continue
            
            robot_traj = wiping_env.compute_feasible_targets_robot_traj()
            if len(robot_traj) <= 1:
                print(f'{arm} valid trajectory not found!')
                continue

            if len(robot_traj) <= 5:
                robot_traj.extend(robot_traj[::-1])

            robot_traj = wiping_env.interpolate_trajectory(robot_traj, alpha=0.5)
            robot_traj = wiping_env.interpolate_trajectory(robot_traj, alpha=0.5)

            # compute feasible targets parameters
            feasible_targets_pos_world, feasible_targets_orn_world, feasible_targets_count, feasible_targets_indices, init_q_R, arm_side = wiping_env.targets_util.get_feasible_targets_lists()
            feasible_targets = manip_env.targets_util.get_feasible_targets_given_indices(feasible_targets_indices, arm_side)
            manip_env.targets_util.set_feasible_targets_lists(feasible_targets_pos_world, feasible_targets_orn_world, feasible_targets, feasible_targets_count, feasible_targets_indices, init_q_R, arm_side)
            manip_env.targets_util.mark_feasible_targets()

            # move robot_2 to wiping initial config
            manip_env.reset_robot(manip_env.robot_2, robot_traj[0])   ####
            manip_env.attach_tool()  ####
            eef_goal_pose = wiping_env.get_eef_pose(robot=wiping_env.robot_2, 
                                                    current_joint_angles=current_joint_angles, target_joint_angles=robot_traj[0])
            # move_robot_loop(manip_env, robot=manip_env.robot_2, other_robot=manip_env.robot, 
            #                 q_robot_init=current_joint_angles, q_robot_goal=robot_traj[0], world_to_robot_eef_goal=eef_goal_pose,
            #                 q_other_robot=q_robot, q_H=q_H)

            # execute wiping trajectory
            for q_R in robot_traj:
                for _ in range(50):
                    manip_env.move_robot(manip_env.robot_2, q_R)
                    manip_env.move_robot(manip_env.robot, q_robot)
                    manip_env.bc.stepSimulation()
                time.sleep(0.05)
                new_target, indices_to_delete = manip_env.targets_util.get_new_contact_points(targeted_arm=arm)
                manip_env.targets_util.remove_contacted_feasible_targets(indices_to_delete, arm)
                wiping_env.targets_util.remove_contacted_feasible_targets(indices_to_delete, arm)

                targets_cleared += new_target
                total_targets_cleared += new_target

            manip_env.targets_util.remove_targets()
            manip_env.targets_util.unmark_feasible_targets()
            manip_env.targets_util.update_targets()
            manip_env.detach_tool()  ####

            wiping_env.targets_util.remove_targets()
            wiping_env.targets_util.update_targets()

            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_2)

    # move robot_2 back to rest poses
    manip_env.reset_robot(manip_env.robot_2, q_robot_2_init)   ####
    eef_goal_pose = wiping_env.get_eef_pose(robot=wiping_env.robot_2, 
                                            current_joint_angles=current_joint_angles, target_joint_angles=q_robot_2_init)
    # move_robot_loop(manip_env, robot=manip_env.robot_2, other_robot=manip_env.robot, 
    #                 q_robot_init=current_joint_angles, q_robot_goal=q_robot_2_init, world_to_robot_eef_goal=eef_goal_pose,
    #                 q_other_robot=q_robot, q_H=q_H)
    wiping_env.reset_robot(wiping_env.robot_2, q_robot_2_init)

    manip_env.unlock_human_joints(q_H)  ####

    print('wiping loop is done')
    return targets_cleared, total_targets_cleared

def arm_manipulation_loop(manip_env, q_robot_2, q_robot_init, q_robot_goal, q_H_init, world_to_eef_goal, q_R_init_traj):
    # Step 0: instantiate a new motion planning problem
    trajectory_planner = manip_env.init_traj_planner(manip_env.world_to_robot_base, clamp_by_human=True,
                                                     q_H_init=q_H_init, q_R_init=q_robot_init)
    trajectory_follower = manip_env.init_traj_follower(manip_env.world_to_robot_base)
    
    # Step 1: move robot to grasping pose
    manip_env.reset_robot(manip_env.robot_2, q_robot_2)
    manip_env.reset_robot(manip_env.robot, q_robot_init)
    manip_env.reset_human_arm(q_H_init)
    manip_env.targets_util.update_targets()

    # Step 2: attach human arm to eef
    env_pcd, right_arm_pcd, _ = manip_env.compute_env_pcd(robot=manip_env.robot_2)
    T_eef_to_object, T_object_to_world = manip_env.attach_human_arm_to_eef(right_arm_pcd, attach_to_gripper=True, trajectory_planner=trajectory_planner)

    # Step 3: trajectory after grasping
    trajectory_planner = manip_env.init_mppi_planner(trajectory_planner, q_robot_init, q_robot_goal, clamp_by_human=True, init_traj=q_R_init_traj)
    traj = manip_env.get_mppi_trajectory(trajectory_planner, q_robot_init)
    previous_update_time = time.time()
    update_second = 5  # sec

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

    trajectory_follower._init_H_clamping(manip_env.eef_to_cp, manip_env.right_elbow_joint_to_cp, manip_env.robot_base_pose,
                                         manip_env.human_arm_lower_limits, manip_env.human_arm_upper_limits, 
                                         human_rest_poses=q_H_init, robot_rest_poses=q_robot_init)

    # Step 5: simulation loop
    while True:
        # if near goal, execute rest of trajectory and end simulation loop
        if manip_env.is_near_goal_W_space(world_to_eef, world_to_eef_goal, threshold=0.05):
            for _ in range(300):
                manip_env.move_robot(manip_env.robot_2, q_robot_2)
                manip_env.move_robot(manip_env.robot, q_robot_goal)
                manip_env.bc.stepSimulation()
            manip_env.targets_util.update_targets()
            break

        # get velocity command
        next_joint_angles = trajectory_follower.follow_trajectory(current_joint_angles, current_human_joint_angles, time_step=0.05)
        current_time = time.time()

        # update trajectory 
        if current_time-previous_update_time > update_second:
            print('replanning...')
            traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles)
            trajectory_follower.update_trajectory(traj)
            previous_update_time = time.time()

        # if valid velocity command, move robot
        else:
            for _ in range(300):
                manip_env.move_robot(manip_env.robot_2, q_robot_2)
                manip_env.move_robot(manip_env.robot, next_joint_angles)
                manip_env.bc.stepSimulation()
            manip_env.targets_util.update_targets()

            # save current_joint_angle
            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
            current_human_joint_angles = manip_env.get_human_joint_angles()
            world_to_eef = manip_env.bc.getLinkState(manip_env.robot.id, manip_env.robot.eef_id)[:2]

    # Step 6: reinforce the grasp
    manip_env.detach_human_arm_from_eef()
    world_to_right_elbow = manip_env.bc.getLinkState(manip_env.humanoid._humanoid, manip_env.right_elbow)[:2]
    world_to_cp = manip_env.bc.multiplyTransforms(world_to_right_elbow[0], world_to_right_elbow[1],
                                                  manip_env.right_elbow_to_cp[0], manip_env.right_elbow_to_cp[1])
    world_to_eef = manip_env.bc.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                                   manip_env.cp_to_eef[0], manip_env.cp_to_eef[1])
    q_R_goal = manip_env.bc.calculateInverseKinematics(manip_env.robot.id, manip_env.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                    manip_env.robot.arm_lower_limits, manip_env.robot.arm_upper_limits, manip_env.robot.arm_joint_ranges, manip_env.robot.arm_rest_poses,
                                                    maxNumIterations=50)
    q_R_goal = [q_R_goal[i] for i in range(len(manip_env.robot.arm_controllable_joints))]
    manip_env.reset_robot(manip_env.robot, q_R_goal)
    manip_env.attach_human_arm_to_eef()

    print('arm manipulation loop is done')

def move_robot_loop(manip_env, robot, other_robot, q_robot_init, q_robot_goal, world_to_robot_eef_goal, q_other_robot, q_H):
    # Step 0: instantiate a new motion planning problem
    if robot == manip_env.robot:
        world_to_robot_base = manip_env.world_to_robot_base
    elif robot == manip_env.robot_2:
        world_to_robot_base = manip_env.world_to_robot_2_base
    else:
        raise ValueError('invalid robot!')
    
    trajectory_planner = manip_env.init_traj_planner(world_to_robot_base, clamp_by_human=False, q_H_init=None, q_R_init=None)
    trajectory_follower = manip_env.init_traj_follower(world_to_robot_base)
    
    # Step 1: initialize trajectory planner & follower
    if manip_env.human_cid is None:
        manip_env.reset_human_arm(q_H)
    trajectory_planner = manip_env.init_mppi_planner(trajectory_planner, q_robot_init, q_robot_goal, clamp_by_human=False, init_traj=[])
    env_pcd, right_arm_pcd, right_shoulder_pcd = manip_env.compute_env_pcd(other_robot)
    env_pcd = np.vstack((env_pcd, right_arm_pcd, right_shoulder_pcd))

    trajectory_planner.update_obstacle_pcd(env_pcd)
    trajectory_follower.update_obstacle_pcd(env_pcd)
    current_joint_angles = q_robot_init
    world_to_eef = manip_env.bc.getLinkState(robot.id, robot.eef_id)[:2]

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
        if manip_env.is_near_goal_W_space(world_to_eef, world_to_robot_eef_goal, threshold=0.1):
            traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles)
            for q_R in traj:
                for _ in range(300):
                    if manip_env.human_cid is None:
                        manip_env.move_human_arm(q_H)
                    manip_env.move_robot(robot, q_R)
                    manip_env.move_robot(other_robot, q_other_robot)
                    manip_env.bc.stepSimulation()
            break

        # get velocity command
        next_joint_angles = trajectory_follower.follow_trajectory(current_joint_angles, current_human_joint_angles=[], time_step=0.05)
        current_time = time.time()

        # update trajectory 
        if current_time-previous_update_time > update_second:
            print('replanning...')
            traj = manip_env.get_mppi_trajectory(trajectory_planner, current_joint_angles)
            trajectory_follower.update_trajectory(traj)
            previous_update_time = time.time()

        # move robot
        else:
            for _ in range(300):
                if manip_env.human_cid is None:
                    manip_env.reset_human_arm(q_H)
                manip_env.move_robot(robot, next_joint_angles)
                manip_env.move_robot(other_robot, q_other_robot)
                manip_env.bc.stepSimulation()

            # save current_joint_angle
            current_joint_angles = manip_env.get_robot_joint_angles(robot)
            world_to_eef = manip_env.bc.getLinkState(robot.id, robot.eef_id)[:2]

    print('move robot loop is done')

if __name__ == '__main__':
    # simulation environments
    wiping_env = WipingDemo()
    manip_env = ManipulationDemo()
    grasp_env = GraspDemo()
    wiping_env.reset()
    manip_env.reset()
    grasp_env.reset()

    manip_env.lock_robot_gripper_joints(manip_env.robot)  ######

    # initial joint states
    q_robot_init = manip_env.robot.arm_rest_poses
    q_robot_2_init = manip_env.robot_2.arm_rest_poses
    q_H_init = manip_env.human_rest_poses

    ### grasp generation
    q_H_init = manip_env.human_rest_poses
    print('generating grasps...')
    # q_R_grasp_samples, grasp_pose_samples, best_q_R_grasp, best_world_to_grasp, best_world_to_eef_goal = grasp_env.generate_grasps(q_H_init)
    # best_q_R_grasp = [-1.18542512, -1.75121252,  2.19404085, -1.71744685, -0.88838092,  0.31547203]
    # best_world_to_grasp = [[0.44916177, 0.34919802, 0.39658533], [ 0.83262646, -0.19758316, -0.48711703,  0.17438771]]
    # best_world_to_eef_goal = ((0.3864336311817169, 0.4190356731414795, 0.5004498362541199), (0.832626461982727, -0.1975831687450409, -0.4871169924736023, 0.17438773810863495))
    best_q_R_grasp = [-2.2567504 , -1.69553655,  2.17958519, -2.02756844, -0.94305021, 0.86691335]
    best_world_to_grasp = [[0.44428981, 0.34869745, 0.39399922], [ 0.84583597, -0.13011431, -0.49919509,  0.13577936]]
    best_world_to_eef_goal = ((0.37870684266090393, 0.39848029613494873, 0.5072271823883057), (0.8458359837532043, -0.13011430203914642, -0.4991950988769531, 0.13577939569950104))

    # save grasp parameters
    manip_env.compute_grasp_parameters(q_H_init, best_q_R_grasp, best_world_to_grasp)
    (right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
     right_elbow_joint_to_cp, cp_to_right_elbow_joint,
     right_wrist_joint_to_cp, cp_to_right_wrist_joint) = manip_env.get_grasp_parameters()
    wiping_env.set_grasp_parameters(right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
                                    right_elbow_joint_to_cp, cp_to_right_elbow_joint,
                                    right_wrist_joint_to_cp, cp_to_right_wrist_joint)
    grasp_env.set_grasp_parameters(right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
                                   right_elbow_joint_to_cp, cp_to_right_elbow_joint,
                                   right_wrist_joint_to_cp, cp_to_right_wrist_joint)

    ### 1st wiping iter (with rest poses)
    success_rate = 0.0
    total_targets_cleared = 0
    total_targets = wiping_env.targets_util.total_target_count
    manip_env.reset_human_arm(q_H_init)

    start_time = time.time()
    targets_cleared, total_targets_cleared = wiping_loop(wiping_env, manip_env, q_H_init, total_targets_cleared, 
                                                         q_robot_init, q_robot_2_init)
    success_rate = total_targets_cleared/total_targets
    print(f'total_targets_cleared: {total_targets_cleared}/{total_targets}')

    # simulation loop until threshold is met...
    current_human_joint_angles = q_H_init
    current_joint_angles = q_robot_init
    target_joint_angles = best_q_R_grasp
    current_robot_2_joint_angles = q_robot_2_init
    for i in range(50):
        # # move robot to grasp pose
        # if manip_env.human_cid is None:
        #     # move to grasp pose for the first arm manipulation
        #     manip_env.reset_human_arm(current_human_joint_angles)
        #     move_robot_loop(manip_env, robot=manip_env.robot, other_robot=manip_env.robot_2, 
        #                     q_robot_init=current_joint_angles, q_robot_goal=target_joint_angles, world_to_robot_eef_goal=best_world_to_eef_goal,
        #                     q_other_robot=current_robot_2_joint_angles, q_H=current_human_joint_angles)
        #     manip_env.reset_human_arm(q_H_init)
        #     manip_env.reset_robot(manip_env.robot, best_q_R_grasp)
        #     current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)

        # reset to grasp pose
        if manip_env.human_cid is None:
            manip_env.reset_human_arm(q_H_init)
            manip_env.reset_robot(manip_env.robot, best_q_R_grasp)
            current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
        
        ### find q_H_goal and q_R_goal using the grasp (human config with best score)
        print('finding for goal configs...')
        q_H_score, q_H_traj, q_R_traj, q_H_goal, q_R_goal = wiping_env.get_best_valid_goal_configs(q_H_init=current_human_joint_angles, 
                                                                                                   q_robot=current_joint_angles, 
                                                                                                   q_robot_2=current_robot_2_joint_angles)
        print(f'q_H score: {q_H_score}, q_H: {q_H_goal}, q_R: {q_R_goal}')

        # save goal parameters
        wiping_env.reset_robot(manip_env.robot, q_R_goal)
        world_to_eef_goal = wiping_env.bc.getLinkState(wiping_env.robot.id, wiping_env.robot.eef_id)[:2]

        ### arm manipulation
        arm_manipulation_loop(manip_env, q_robot_2=current_robot_2_joint_angles, 
                              q_robot_init=current_joint_angles, q_robot_goal=q_R_goal, q_H_init=current_human_joint_angles,
                              world_to_eef_goal=world_to_eef_goal,
                              q_R_init_traj=q_R_traj)
        
        current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
        current_human_joint_angles = manip_env.get_human_joint_angles()

        # ### no arm manipulation (just use reset)
        # for q_H, q_R in zip(q_H_traj, q_R_traj):
        #     manip_env.reset_human_arm(q_H)
        #     manip_env.reset_robot(manip_env.robot, q_R)
        #     time.sleep(0.5)
        # current_human_joint_angles = manip_env.get_human_joint_angles()
        # current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)

        ### n-th wiping iter
        targets_cleared, total_targets_cleared = wiping_loop(wiping_env, manip_env, current_human_joint_angles, total_targets_cleared, 
                                                             q_robot=current_joint_angles, q_robot_2_init=current_robot_2_joint_angles)

        # check if wiping threshold is reached
        success_rate = total_targets_cleared/total_targets
        print(f'iter {i+1} | success_rate: {success_rate}, new targets cleared: {targets_cleared}, total_targets_cleared: {total_targets_cleared}/{total_targets}')
        if success_rate >= 0.8:
            break

        # save states
        current_human_joint_angles = manip_env.get_human_joint_angles()
        current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
        current_robot_2_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot_2)
    
    # end of the simulation loop
    total_time = time.time() - start_time
    print(f'success_rate: {success_rate}, total_targets_cleared: {total_targets_cleared}/{total_targets}')
    print(f'iteration: {i}, total simulation time: {total_time}')
    print('done')
