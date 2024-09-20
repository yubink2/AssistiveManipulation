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
    env_pcd, right_arm_pcd, _ = manip_env.compute_env_pcd(manip_env.robot_2)
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

    #### DEBGIGGJUGUGGU LIFEAFJSDLKFJ
    manip_env.detach_human_arm_from_eef()
    for q_R in traj:
        # for _ in range(500):
            # manip_env.move_robot(manip_env.robot_2, q_robot_2)
            # manip_env.move_robot(manip_env.robot, q_R)
            # manip_env.bc.stepSimulation()
        manip_env.reset_robot(manip_env.robot_2, q_robot_2)
        manip_env.reset_robot(manip_env.robot, q_R)
        manip_env.reset_human_arm(manip_env.get_q_H_from_eef_pose())
        manip_env.bc.stepSimulation()
        manip_env.targets_util.update_targets()
        time.sleep(0.1)
    print('here')

    manip_env.reset_robot(manip_env.robot, q_robot_init)
    manip_env.reset_human_arm(q_H_init)
    manip_env.attach_human_arm_to_eef()
    ####

    # Step 5: simulation loop
    while True:
        # if near goal, execute rest of trajectory and end simulation loop
        if manip_env.is_near_goal_W_space(world_to_eef, world_to_eef_goal, threshold=0.05):
            for _ in range(500):
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
            for _ in range(500):
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

if __name__ == '__main__':
    # simulation environments
    wiping_env = WipingDemo()
    manip_env = ManipulationDemo()
    grasp_env = GraspDemo()
    wiping_env.reset()
    manip_env.reset()
    grasp_env.reset()

    # initial joint states
    q_robot_init = manip_env.robot.arm_rest_poses
    q_robot_2_init = manip_env.robot_2.arm_rest_poses
    q_H_init = manip_env.human_rest_poses

    #### DEBUGGING
    # configs to be tested
    # q_R_grasp_samples, grasp_pose_samples, best_q_R_grasp, best_world_to_grasp, best_world_to_eef_goal = grasp_env.generate_grasps(q_H_init)
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

    # q_H_goal = [2.7161094340709035, 0.06622645631378093, -2.188650490192958, 1.5295333010953591]
    # world_to_right_elbow = ((0.32202110721651805, 0.3101278503233666, 0.4711766969965591), (-0.2619213717538959, -0.5299630650847919, 0.7916604948076205, 0.15430491121621392))
    # q_H_goal = [1.5900651151776706, 0.6221374109686765, -1.8872283515668193, 0.415438256408006]
    # world_to_right_elbow = ((0.36141975958199357, 0.5837208659430699, 0.4191382241840883), (-0.09718898526710897, -0.13158468293383416, 0.951359519383235, 0.26106481424897526))

    # q_H_goal = [1.7233638137971163, -0.1413451915070879, -2.4539169162731995, 0.9360700272505206] #works
    # q_H_goal = [2.9278844169229754, 0.7419305355181721, -1.9232331502579587, 0.509418540858362] #works
    # q_H_goal = [2.9648957635739523, 0.09395099975365467, -2.473966244785029, 1.5817405944984517] #works
    # q_H_goal = [2.365488141186549, 0.5348255460180686, -2.4621561739544124, 0.6115218078794264] #works
    # q_H_goal = [-2.981728680747616, 0.5791305894707974, -2.1246894967472967, 0.5160677743733465] #works, but q_H_traj looking weird
    # q_H_goal = [3.1165447972854428, 0.3207504409269406, -2.3102116978896, 0.8639247399021508]
    # q_H_goal = [-2.670614696121192, -0.15765899661718902, -2.368976882138696, 1.2615248796869514]
    # q_H_goal = [2.4317462853796976, -0.06690456644567999, -2.513894718212701, 1.2506467421008278]
    q_H_goal = [2.54526294401084, -0.09419138532833907, -2.5356874992464844, 1.3103610219078765]


    # compute initial trajectories
    q_H_traj, q_R_traj = manip_env.get_init_traj_from_q_H(q_H_init=q_H_init, q_H_goal=q_H_goal, q_R_init=best_q_R_grasp)
    # q_H_traj = [[2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409], [2.4956259226779016, -0.03586514687057306, -1.993552810254121, 0.6672795393810498], [2.5121715964555476, -0.055307226356495064, -2.1742643732515754, 0.8816400335566588], [2.5287172702331935, -0.07474930584241707, -2.35497593624903, 1.0960005277322677], [2.54526294401084, -0.09419138532833907, -2.5356874992464844, 1.3103610219078765]]
    # q_R_traj = [[1.4400192410260368, -1.8322057422692213, -1.6836236283643051, -3.992126657400728, -2.1065423373831087, -1.6141345040719464], [1.303333348623088, -1.930360192678072, -1.553599034910307, -4.024003625258252, -2.3874607044419487, -1.6268691198474385], [1.2061703828983272, -2.008496679617555, -1.4732180352324389, -3.932617364972397, -2.646716651947915, -1.5376453689672267], [1.1274393008720414, -2.029151657892788, -1.4534211143626103, -3.5399609788130033, -2.8727999183174053, -1.161045185798169], [1.0201701328617578, -1.9534271867400654, -1.5719016154271155, -2.595652963693605, -2.9849462086517398, -0.29083600121737097]]

    for q_H, q_R in zip(q_H_traj, q_R_traj):
        manip_env.reset_human_arm(q_H)
        manip_env.reset_robot(manip_env.robot, q_R)
        time.sleep(0.5)

    # save goal parameters
    q_R_goal = q_R_traj[-1]
    manip_env.reset_robot(manip_env.robot, q_R_goal)
    world_to_eef_goal = manip_env.bc.getLinkState(manip_env.robot.id, manip_env.robot.eef_id)[:2]

    # check if valid?
    manip_env.reset_human_arm(q_H_goal)
    world_to_right_elbow = manip_env.bc.getLinkState(manip_env.humanoid._humanoid, manip_env.right_elbow)[:2]
    print('are goal configs valid? ', wiping_env.validate_q_R_goal(q_H_goal, world_to_right_elbow, q_R_goal))

    # reset to grasp pose
    manip_env.reset_human_arm(q_H_init)
    manip_env.reset_robot(manip_env.robot, best_q_R_grasp)
    current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
    
    print('human goal parameters: ', q_H_goal)
    print('robot goal parameters: ', q_R_goal, world_to_eef_goal)
    arm_manipulation_loop(manip_env, q_robot_2=q_robot_2_init, 
                            q_robot_init=current_joint_angles, q_robot_goal=q_R_goal, q_H_init=q_H_init,
                            world_to_eef_goal=world_to_eef_goal, q_R_init_traj=q_R_traj)
    current_joint_angles = manip_env.get_robot_joint_angles(manip_env.robot)
    current_human_joint_angles = manip_env.get_human_joint_angles()

    sys.exit()
    ####
