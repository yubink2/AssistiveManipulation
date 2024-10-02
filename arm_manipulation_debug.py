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

    # configs to be tested
    # q_R_grasp_samples, grasp_pose_samples, best_q_R_grasp, best_world_to_grasp, best_world_to_eef_goal = grasp_env.generate_grasps(q_H_init)
    best_q_R_grasp = [-2.2567504 , -1.69553655,  2.17958519, -2.02756844, -0.94305021, 0.86691335]
    best_world_to_grasp = [[0.44428981, 0.34869745, 0.39399922], [ 0.84583597, -0.13011431, -0.49919509,  0.13577936]]
    best_world_to_eef_goal = ((0.37870684266090393, 0.39848029613494873, 0.5072271823883057), (0.8458359837532043, -0.13011430203914642, -0.4991950988769531, 0.13577939569950104))
    q_R_init = best_q_R_grasp

    ###########

    manip_env.reset_robot(manip_env.robot, best_q_R_grasp)
    for j in range(manip_env.bc.getNumJoints(manip_env.robot.id, physicsClientId=manip_env.bc._client)):
        print(manip_env.bc.getJointState(manip_env.robot.id, j)[0])

    ###########

    ## 1
    q_H_init = [1.9568880587220205, -0.0423682199434994, -2.1238626018172133, 0.7425848680343998]
    q_R_init = [-2.6824140887233145, -1.7287522015048118, 2.11173299930017, -0.9794660164864212, -0.38799615527359926, -0.35254809279265376]
    q_H_goal = [2.4611092674613873, 0.23041666252273935, -2.387197180190736, 0.7005500175358801]

    ## 2
    q_H_init = [2.6658202676762675, 0.9631405213180286, -2.2353058583300403, 0.7360792909939619]
    q_R_init = [-2.5542882386149683, -0.8025827162244286, 0.8166208882768811, 0.25710220222200625, -0.9615371929300279, -0.14765332288502034]
    q_H_goal = [0.9554150769980202, 0.22470739369179193, -2.016637261917095, 0.7009740835333081]

    ## 3
    q_H_init = [2.9684811805455844, 0.1483789117555756, -2.332427714060345, 0.6247270997086152]
    q_R_init = [-2.2742741954595496, -1.1784769285493595, 1.4490840574112702, -1.4190784327104702, -0.7579316829582062, 1.2603950829201205]
    q_H_goal = [1.785724098906827, 0.07822929468215778, -2.5986329943934843, 0.5351548133071207]

    # ## 4
    # best_q_R_grasp = [-2.31536267, -1.71909261,  2.42871111, -2.59149997, -0.73822019,  1.06473343]
    # best_world_to_grasp = [[0.4455078 , 0.34882259, 0.39464574], [ 0.8105788 , -0.2775272 , -0.4689705 ,  0.21449321]]
    # best_world_to_eef_goal = ((0.3885570466518402, 0.43996644020080566, 0.48436439037323), (0.8105787634849548, -0.27752718329429626, -0.46897047758102417, 0.21449322998523712))
    # q_H_init = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]
    # q_R_init = [-2.3153626669183445, -1.7190926054548499, 2.428711113527493, -2.5914999700013475, -0.7382201912375622, 1.0647334254010368]
    # q_H_goal = [2.1412306200289386, -0.23419611235843588, -2.4450768567046217, 0.08961029658521918]
    # q_R_goal = [-2.1456190271699196, -1.3286291962538042, 1.9929990805508555, -3.0001441409304768, -1.0929089595544932, 2.035660567591221]

    ## 5
    q_H_init = [2.5507480523731147, 0.013283697292815562, -1.9226861094256793, 0.5131182519857714]
    q_R_init = [-2.297016338696575, -1.6344301472823213, 2.0333724033317258, -1.8027892025749779, -0.8755342390033943, 0.8519503788750441]
    q_H_goal = [2.1317303844598157, 0.2789572153997856, -2.1949948343941283, 1.1988104618815019]

    ## 6
    q_H_init = [2.2484613051810607, 0.47843743351063395, -2.3447310983239555, 0.42231815228520786]
    q_R_init = [-2.607720347731264, -1.4418791792244832, 1.5396282700831223, -0.6344648462007815, -0.6309698947044574, 0.1506821862248122]
    q_H_goal = [1.0438756747561124, -0.15867169422582506, -2.161962715403517, 0.5813161938859222]

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

    # compute initial trajectories
    q_H_traj, q_R_traj = manip_env.get_init_traj_from_q_H(q_H_init=q_H_init, q_H_goal=q_H_goal, q_R_init=q_R_init)

    for q_H, q_R in zip(q_H_traj, q_R_traj):
        manip_env.reset_human_arm(q_H)
        manip_env.reset_robot(manip_env.robot, q_R)
        print('are goal configs valid? ', wiping_env.validate_q_R(q_H, q_R))
        time.sleep(0.3)

    # save goal parameters
    q_R_goal = q_R_traj[-1]
    manip_env.reset_robot(manip_env.robot, q_R_goal)
    world_to_eef_goal = manip_env.bc.getLinkState(manip_env.robot.id, manip_env.robot.eef_id)[:2]

    # check if valid?
    manip_env.reset_human_arm(q_H_goal)
    world_to_right_elbow = manip_env.bc.getLinkState(manip_env.humanoid._humanoid, manip_env.right_elbow)[:2]

    # reset to grasp pose
    manip_env.reset_human_arm(q_H_init)
    manip_env.reset_robot(manip_env.robot, q_R_init)
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
