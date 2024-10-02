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

# utils
from utils.collision_utils import get_collision_fn
from wiping_task.util import Util
from wiping_task.targets_util import TargetsUtil


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


class ManipulationDemo():
    def __init__(self):
        # Start the bullet physics server
        self.bc = BulletClient(connection_mode=p.GUI)
        # self.bc = BulletClient(connection_mode=p.DIRECT)
        self.bc.setPhysicsEngineParameter(numSolverIterations=200)

        self.util = Util(self.bc._client)
        self.targets_util = TargetsUtil(self.bc._client, self.util)

    def reset(self):
        self.create_world()
        self.init_tool()

        # get 'static' obstacle point cloud
        self.static_obstacles = [self.bed_id, self.cube_id, self.cube_2_id]
        self.static_obs_pcd = self.get_obstacle_point_cloud(self.static_obstacles)

        ### wiping robot parameters
        # initialize collision checker
        robot_2_obstacles = [self.bed_id, self.humanoid._humanoid]
        self.robot_2_in_collision = get_collision_fn(self.robot_2.id, self.robot_2.arm_controllable_joints, obstacles=robot_2_obstacles,
                                                     attachments=[], self_collisions=True,
                                                     disabled_collisions=set(), client_id=self.bc._client)
        
        # compute target_to_eef & target_closer_to_eef
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        target_orn = self.util.rotate_quaternion_by_axis(world_to_eef[1], axis='z', degrees=180)
        world_to_target = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.3], target_orn]
        world_to_target_closer = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.18], target_orn]

        self.target_orn = target_orn
        target_to_world = self.bc.invertTransform(world_to_target[0], world_to_target[1], physicsClientId=self.bc._client)
        target_closer_to_world = self.bc.invertTransform(world_to_target_closer[0], world_to_target_closer[1], physicsClientId=self.bc._client)
        self.target_to_eef = self.bc.multiplyTransforms(target_to_world[0], target_to_world[1],
                                                        world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        self.target_closer_to_eef = self.bc.multiplyTransforms(target_closer_to_world[0], target_closer_to_world[1],
                                                               world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)

        # generate targets
        self.targets_util.init_targets_util(self.humanoid._humanoid, self.right_shoulder, self.right_elbow, self.human_right_arm,
                                            self.robot_2, self.tool,
                                            self.target_to_eef, self.target_closer_to_eef, self.robot_2_in_collision)
        self.targets_util.generate_new_targets_pose()
        self.targets_util.generate_targets()
        self.targets_util.initialize_deleted_targets_list()

    def create_world(self):
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.bc.setGravity(0, 0, -9.8) 
        self.bc.setGravity(0, 0, 0)
        self.bc.setTimestep = 0.005

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, 0, -0.04))
        self.bed_id = self.bc.loadURDF("./urdf/bed_0.urdf", (0.0, -0.1, 0.0), useFixedBase=True)
        self.human_cid = None
        self.tool_cid = None

        # load human
        human_base_pos = (0, 0, 0.3)
        human_base_orn = self.bc.getQuaternionFromEuler((0, 1.57, 0))
        motionPath = 'deep_mimic/mocap/data/Sitting1.json'
        self.motion = MotionCaptureData()
        self.motion.Load(motionPath)
        self.humanoid = Humanoid(self.bc, self.motion, baseShift=human_base_pos, ornShift=human_base_orn)
        
        self.right_shoulder = 6
        self.right_elbow = 7
        self.right_wrist = 8
        self.human_controllable_joints = [3, 4, 5, 7]
        self.human_right_arm = [3, 4, 5, 6, 7, 8]
        self.human_rest_poses = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]

        human_base = self.bc.getBasePositionAndOrientation(self.humanoid._humanoid)[:2]
        self.T_world_to_human_base = compute_matrix(translation=human_base[0], rotation=human_base[1])

        # load first robot (manipulation)
        self.robot_base_pose = ((0.5, 0.8, 0.25), (0, 0, 0))
        self.cube_id = self.bc.loadURDF("./urdf/cube_0.urdf", 
                                   (self.robot_base_pose[0][0], self.robot_base_pose[0][1], self.robot_base_pose[0][2]-0.15), useFixedBase=True)
        self.world_to_robot_base = compute_matrix(translation=self.robot_base_pose[0], rotation=self.robot_base_pose[1], rotation_type='euler')
        self.robot = UR5Robotiq85(self.bc, self.robot_base_pose[0], self.robot_base_pose[1])
        self.robot.load()
        for _ in range(50):
            self.robot.reset()
            self.robot.open_gripper()

        # load second robot (wiping)
        self.robot_2_base_pose = ((0.55, 0, 0), (0, 0, -1.57))
        self.cube_2_id = self.bc.loadURDF("./urdf/cube_0.urdf", 
                            (self.robot_2_base_pose[0][0], self.robot_2_base_pose[0][1], self.robot_2_base_pose[0][2]-0.15), useFixedBase=True)
        self.world_to_robot_2_base = compute_matrix(translation=self.robot_2_base_pose[0], rotation=self.robot_2_base_pose[1], rotation_type='euler')
        self.robot_2 = UR5Robotiq85(self.bc, self.robot_2_base_pose[0], self.robot_2_base_pose[1])
        self.robot_2.load()
        self.robot_2.reset()

        # initialize robot parameters
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        world_to_eef_grasp = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.14],
                              world_to_eef[1]]
        eef_grasp_to_world = self.bc.invertTransform(world_to_eef_grasp[0], world_to_eef_grasp[1])
        eef_grasp_to_eef = self.bc.multiplyTransforms(eef_grasp_to_world[0], eef_grasp_to_world[1],
                                                      world_to_eef[0], world_to_eef[1])
        self.eef_grasp_to_eef = eef_grasp_to_eef

        # initialize collision checker        
        robot_obstacles = [self.bed_id, self.robot_2.id, self.cube_2_id, self.humanoid._humanoid]
        self.robot_in_collision = get_collision_fn(self.robot.id, self.robot.arm_controllable_joints, obstacles=robot_obstacles,
                                                   attachments=[], self_collisions=True,
                                                   disabled_collisions=set(), client_id=self.bc._client)
        
        ### DEUBUGGING
        self.robot_in_collision_debug = get_collision_fn(self.robot.id, self.robot.arm_controllable_joints, obstacles=[],
                                                   attachments=[], self_collisions=True,
                                                   disabled_collisions=set(), client_id=self.bc._client)
        
        # initialize human parameters
        shoulder_min = [-3.141368118925281, -0.248997453133789, -2.6643015908664056]  # order: [yaw, pitch, roll]
        shoulder_max = [3.1415394736319917, 1.2392816988875348, -1.3229245882839409]  # order: [yaw, pitch, roll]
        elbow_min = [0.0]
        # elbow_min = [0.401146]
        elbow_max = [2.541304]
        self.human_arm_lower_limits = shoulder_min + elbow_min
        self.human_arm_upper_limits = shoulder_max + elbow_max
        self.human_arm_joint_ranges = list(np.array(self.human_arm_upper_limits) - np.array(self.human_arm_lower_limits))

    def init_tool(self):
        # close gripper
        for _ in range(50):
            self.robot_2.move_gripper(0.04)
            self.bc.stepSimulation(physicsClientId=self.bc._client)

        # initialize tool for wiping task
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        world_to_tool = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.14], [0,0,0,1]]
        eef_to_tool = self.bc.multiplyTransforms(positionA=eef_to_world[0], orientationA=eef_to_world[1], 
                                                 positionB=world_to_tool[0], orientationB=world_to_tool[1], physicsClientId=self.bc._client)
        self.eef_to_tool = eef_to_tool
        self.tool = self.bc.loadURDF("./urdf/wiper.urdf", basePosition=world_to_tool[0], baseOrientation=world_to_tool[1], physicsClientId=self.bc._client)

        # disable collisions between the tool and robot
        for j in self.robot_2.arm_controllable_joints:
            for tj in list(range(self.bc.getNumJoints(self.tool, physicsClientId=self.bc._client))) + [-1]:
                self.bc.setCollisionFilterPair(self.robot_2.id, self.tool, j, tj, False, physicsClientId=self.bc._client)

        # disable collisions between the tool and human arm
        for j in self.human_right_arm:
            for tj in list(range(self.bc.getNumJoints(self.tool, physicsClientId=self.bc._client))) + [-1]:
                self.bc.setCollisionFilterPair(self.humanoid._humanoid, self.tool, j, tj, False, physicsClientId=self.bc._client)

        # disable collisions between the second robot (wiping) and human arm
        for j in self.human_right_arm:
            for tj in list(range(self.bc.getNumJoints(self.robot_2.id, physicsClientId=self.bc._client))):
                self.bc.setCollisionFilterPair(self.humanoid._humanoid, self.robot_2.id, j, tj, False, physicsClientId=self.bc._client)

    def compute_grasp_parameters(self, q_H, q_R_grasp, grasp):
        # compute right_elbow_to_cp
        self.reset_human_arm(q_H)
        world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        world_to_cp = (grasp[0], world_to_right_elbow[1])
        right_elbow_to_world = self.bc.invertTransform(world_to_right_elbow[0], world_to_right_elbow[1])
        right_elbow_to_cp = self.bc.multiplyTransforms(right_elbow_to_world[0], right_elbow_to_world[1],
                                                       world_to_cp[0], world_to_cp[1])
        cp_to_right_elbow = self.bc.invertTransform(right_elbow_to_cp[0], right_elbow_to_cp[1])

        # compute right_elbow_joint_to_cp
        world_to_right_elbow_joint = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[4:6]
        right_elbow_joint_to_world = self.bc.invertTransform(world_to_right_elbow_joint[0], world_to_right_elbow_joint[1])
        right_elbow_joint_to_cp = self.bc.multiplyTransforms(right_elbow_joint_to_world[0], right_elbow_joint_to_world[1],
                                                             world_to_cp[0], world_to_cp[1])
        T_right_elbow_joint_to_cp = compute_matrix(translation=right_elbow_joint_to_cp[0], rotation=right_elbow_joint_to_cp[1])

        # compute right_wrist_joint_to_cp
        world_to_right_wrist_joint = self.bc.getLinkState(self.humanoid._humanoid, self.right_wrist)[4:6]
        right_wrist_joint_to_world = self.bc.invertTransform(world_to_right_wrist_joint[0], world_to_right_wrist_joint[1])
        right_wrist_joint_to_cp = self.bc.multiplyTransforms(right_wrist_joint_to_world[0], right_wrist_joint_to_world[1],
                                                             world_to_cp[0], world_to_cp[1])
        
        # compute eef_to_cp
        self.reset_robot(self.robot, q_R_grasp)
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1])
        eef_to_cp = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                               world_to_cp[0], world_to_cp[1])
        cp_to_eef = self.bc.invertTransform(eef_to_cp[0], eef_to_cp[1])
        
        self.right_elbow_to_cp = right_elbow_to_cp
        self.cp_to_right_elbow = cp_to_right_elbow
        self.T_right_elbow_joint_to_cp = T_right_elbow_joint_to_cp
        self.eef_to_cp = eef_to_cp
        self.cp_to_eef = cp_to_eef

        self.right_elbow_joint_to_cp = right_elbow_joint_to_cp
        self.cp_to_right_elbow_joint = self.bc.invertTransform(right_elbow_joint_to_cp[0], right_elbow_joint_to_cp[1])

        self.right_wrist_joint_to_cp = right_wrist_joint_to_cp
        self.cp_to_right_wrist_joint = self.bc.invertTransform(right_wrist_joint_to_cp[0], right_wrist_joint_to_cp[1])

    def get_grasp_parameters(self):
        return (self.right_elbow_to_cp, self.cp_to_right_elbow,
                self.eef_to_cp, self.cp_to_eef,
                self.right_elbow_joint_to_cp, self.cp_to_right_elbow_joint,
                self.right_wrist_joint_to_cp, self.cp_to_right_wrist_joint)
        
    def compute_env_pcd(self, robot):
        link_to_separate = [self.right_elbow, self.right_wrist]
        human_pcd, separate_pcd = get_humanoid_point_cloud(self.humanoid._humanoid, link_to_separate, client_id=self.bc._client)
        robot_pcd = self.get_robot_point_cloud(robot)

        env_pcd = np.vstack((self.static_obs_pcd, robot_pcd, human_pcd))
        right_arm_pcd = np.array(separate_pcd)
        right_shoulder_pcd = get_point_cloud_from_collision_shapes_specific_link(self.humanoid._humanoid, self.right_shoulder, resolution=8, client_id=self.bc._client)

        return env_pcd, right_arm_pcd, right_shoulder_pcd

    ### INITIALIZE PLANNER
    def init_traj_planner(self, world_to_robot_base, clamp_by_human, q_H_init=None, q_R_init=None):
        JOINT_LIMITS = [
            np.array(self.robot.arm_lower_limits), 
            np.array(self.robot.arm_upper_limits)
        ]

        # Instantiate mppi H clamp
        if clamp_by_human:
            assert q_H_init is not None
            assert q_R_init is not None
            mppi_H_clamp = MPPI_H_Clamp(self.eef_to_cp, self.right_elbow_joint_to_cp, self.robot_base_pose,
                                        self.human_arm_lower_limits, self.human_arm_upper_limits, 
                                        human_rest_poses=q_H_init, robot_rest_poses=q_R_init)
        else:
            mppi_H_clamp = None

        # Instantiate trajectory planner
        trajectory_planner = TrajectoryPlanner(
            joint_limits=JOINT_LIMITS,
            robot_urdf_location=robot_urdf_location,
            scene_urdf_location=scene_urdf_location,
            link_fixed=LINK_FIXED,
            link_ee=LINK_EE,
            link_skeleton=LINK_SKELETON,
            control_points_location = control_points_location,
            control_points_number = control_points_number,
            mppi_H_clamp = mppi_H_clamp,
            world_to_robot_base = world_to_robot_base,
        )
        print("Instantiated trajectory planner")

        return trajectory_planner

    def init_mppi_planner(self, trajectory_planner, current_joint_angles, target_joint_angles, clamp_by_human, init_traj=[]):
        # MPPI parameters
        N_JOINTS = len(self.robot.arm_controllable_joints)
        mppi_control_limits = [
            -0.05 * np.ones(N_JOINTS),
            0.05 * np.ones(N_JOINTS)
        ]
        mppi_covariance = 0.005
        mppi_nsamples = 500
        mppi_lambda = 0.5

        # Update whether to clamp_by_human
        trajectory_planner.update_clamp_by_human(clamp_by_human)

        # Instantiate MPPI object
        trajectory_planner.instantiate_mppi_ja_to_ja(
            current_joint_angles,
            target_joint_angles,
            init_traj=init_traj,
            mppi_control_limits=mppi_control_limits,
            mppi_nsamples=mppi_nsamples,
            mppi_covariance=mppi_covariance,
            mppi_lambda=mppi_lambda,
            waypoint_density = 5,
            # action_smoothing= 0.3,
        )
        print('Instantiated MPPI object')

        return trajectory_planner

    def get_mppi_trajectory(self, trajectory_planner, current_joint_angles):
        # Plan trajectory
        start_time = time.time()
        trajectory = trajectory_planner.get_mppi_rollout(current_joint_angles)
        print("planning time : ", time.time()-start_time)
        # print(np.array(trajectory))
        return trajectory
    
    def init_traj_follower(self, world_to_robot_base):
        JOINT_LIMITS = [
            np.array(self.robot.arm_lower_limits), 
            np.array(self.robot.arm_upper_limits)
        ]

        # Trajectory Follower initialization
        trajectory_follower = TrajectoryFollower(
            joint_limits = JOINT_LIMITS,
            robot_urdf_location = robot_urdf_location,
            control_points_json = control_points_location,
            link_fixed = LINK_FIXED,
            link_ee = LINK_EE,
            link_skeleton = LINK_SKELETON,
            control_points_number = control_points_number,
            world_to_robot_base = world_to_robot_base,
        )
        print('trajectory follower instantiated')

        return trajectory_follower

    def attach_human_arm_to_eef(self, right_arm_pcd=None, joint_type=p.JOINT_FIXED, attach_to_gripper=False, trajectory_planner=None):
        # attach human arm (obj) to eef (body)
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        world_to_eef_base = self.bc.getLinkState(self.robot.id, self.robot.eef_base_id)[:2]
        world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        world_to_right_wrist = self.bc.getLinkState(self.humanoid._humanoid, self.right_wrist)[:2]
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1])
        eef_base_to_world = self.bc.invertTransform(world_to_eef_base[0], world_to_eef_base[1])
        eef_to_right_elbow = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                                        world_to_right_elbow[0], world_to_right_elbow[1])
        eef_to_right_wrist = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                                        world_to_right_wrist[0], world_to_right_wrist[1])
        eef_base_to_right_elbow = self.bc.multiplyTransforms(eef_base_to_world[0], eef_base_to_world[1],
                                                        world_to_right_elbow[0], world_to_right_elbow[1])
        # eef_base_to_right_wrist = self.bc.multiplyTransforms(eef_base_to_world[0], eef_base_to_world[1],
        #                                                 world_to_right_wrist[0], world_to_right_wrist[1])
        
        world_to_left_inner_finger_pad = self.bc.getLinkState(self.robot.id, 12)
        world_to_right_inner_finger_pad = self.bc.getLinkState(self.robot.id, 17)
        left_inner_finger_pad_to_world = self.bc.invertTransform(world_to_left_inner_finger_pad[0], world_to_left_inner_finger_pad[1])
        right_inner_finger_pad_to_world = self.bc.invertTransform(world_to_right_inner_finger_pad[0], world_to_right_inner_finger_pad[1])
        lifp_to_right_elbow = self.bc.multiplyTransforms(left_inner_finger_pad_to_world[0], left_inner_finger_pad_to_world[1],
                                                         world_to_right_elbow[0], world_to_right_elbow[1])
        rifp_to_right_elbow = self.bc.multiplyTransforms(right_inner_finger_pad_to_world[0], right_inner_finger_pad_to_world[1],
                                                         world_to_right_elbow[0], world_to_right_elbow[1])

        if self.right_elbow_to_cp is None or self.eef_to_cp is None:
            raise ValueError('right_elbow_to_cp or eef_to_cp not initialized.')
        
        if self.human_cid is None:
            self.human_cid = self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
                                                    parentLinkIndex=self.robot.eef_id,
                                                    childBodyUniqueId=self.humanoid._humanoid,
                                                    childLinkIndex=self.right_elbow,
                                                    jointType=p.JOINT_FIXED,
                                                    jointAxis=(0, 0, 0),
                                                    parentFramePosition=eef_to_right_elbow[0],
                                                    parentFrameOrientation=eef_to_right_elbow[1],
                                                    childFramePosition=(0, 0, 0),
                                                    childFrameOrientation=(0, 0, 0))
            # self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
            #                                         parentLinkIndex=self.robot.eef_id,
            #                                         childBodyUniqueId=self.humanoid._humanoid,
            #                                         childLinkIndex=self.right_wrist,
            #                                         jointType=p.JOINT_POINT2POINT,
            #                                         jointAxis=(0, 0, 0),
            #                                         parentFramePosition=eef_to_right_wrist[0],
            #                                         parentFrameOrientation=eef_to_right_wrist[1],
            #                                         childFramePosition=(0, 0, 0),
            #                                         childFrameOrientation=(0, 0, 0))

        if attach_to_gripper:
            assert trajectory_planner is not None
            assert right_arm_pcd is not None

            # compute transform matrix from robot's gripper to object frame
            world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
            world_to_cp = self.bc.multiplyTransforms(world_to_right_elbow[0], world_to_right_elbow[1],
                                                     self.right_elbow_to_cp[0], self.right_elbow_to_cp[1])
            T_eef_to_object = compute_matrix(translation=self.eef_to_cp[0], rotation=self.eef_to_cp[1], rotation_type='quaternion')

            # compute transform matrix for inverse of object pose in world frame
            T_world_to_object = compute_matrix(translation=world_to_cp[0], rotation=world_to_cp[1], rotation_type='quaternion')
            T_object_to_world = inverse_matrix(T_world_to_object)

            trajectory_planner.attach_to_gripper(object_type="pcd", object_geometry=right_arm_pcd,
                                                 T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world)

            return T_eef_to_object, T_object_to_world
    
    def detach_human_arm_from_eef(self):
        if self.human_cid is not None:
            self.bc.removeConstraint(self.human_cid)
        self.human_cid = None

    def attach_tool(self):
        # reset tool and attach it to eef
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        world_to_tool = self.bc.multiplyTransforms(world_to_eef[0], world_to_eef[1],
                                                self.eef_to_tool[0], self.eef_to_tool[1], physicsClientId=self.bc._client)
        self.bc.resetBasePositionAndOrientation(self.tool, world_to_tool[0], world_to_tool[1], physicsClientId=self.bc._client)

        # create constraint that keeps the tool in the gripper
        self.tool_cid = self.bc.createConstraint(parentBodyUniqueId=self.robot_2.id,
                            parentLinkIndex=self.robot_2.eef_id,
                            childBodyUniqueId=self.tool,
                            childLinkIndex=-1,
                            jointType=p.JOINT_FIXED,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=self.eef_to_tool[0],
                            parentFrameOrientation=self.eef_to_tool[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0),
                            physicsClientId=self.bc._client)
        
    def detach_tool(self):
        if self.tool_cid is not None:
            self.bc.removeConstraint(self.tool_cid)
        self.tool_cid = None
        self.bc.resetBasePositionAndOrientation(self.tool, [100,100,100], [0,0,0,1], physicsClientId=self.bc._client)

    def visualize_point_cloud(self, pcd):
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([pc_ply])

    def reset_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, joint_id, q_robot[i])
        for j in range(robot.eef_base_id, self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            self.bc.resetJointState(robot.id, j, 0.0)

    def move_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_robot[i])
        for j in range(robot.eef_base_id, self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            self.bc.setJointMotorControl2(robot.id, j, p.POSITION_CONTROL, 0.0)

    def reset_human_arm(self, q_human):
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, j, q_human[i], physicsClientId=self.bc._client)

    def move_human_arm(self, q_human):
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.setJointMotorControl2(self.humanoid._humanoid, j, p.POSITION_CONTROL, q_human[i], physicsClientId=self.bc._client)

    def get_obstacle_point_cloud(self, obstacles):
        point_cloud = []
        for obstacle in obstacles:
            if obstacle == self.bed_id:
                half_extents = [0.5, 1.7, 0.2]
                point_cloud.extend(get_point_cloud_from_collision_shapes(obstacle, half_extents, client_id=self.bc._client))
            else:
                point_cloud.extend(get_point_cloud_from_visual_shapes(obstacle, client_id=self.bc._client))
            # point_cloud.extend(get_point_cloud_from_visual_shapes(obstacle, client_id=self.bc._client))
        return np.array(point_cloud)

    def get_robot_point_cloud(self, robot):
        robot_pcd = get_point_cloud_from_collision_shapes(robot.id, client_id=self.bc._client)
        upper_arm_link = self.bc.getLinkState(robot.id, 2)[:2]
        forearm_link = self.bc.getLinkState(robot.id, 3)[:2]
        upper_arm_pcd = generate_capsule_vertices(radius=0.04, height=0.3, position=upper_arm_link[0], 
                                                  orientation=upper_arm_link[1], client_id=self.bc._client)
        forearm_pcd = generate_capsule_vertices(radius=0.04, height=0.27, position=forearm_link[0], 
                                                orientation=forearm_link[1], client_id=self.bc._client)
        pcd = np.vstack((robot_pcd, upper_arm_pcd, forearm_pcd))
        return pcd

    def is_near_goal_C_space(self, current_joint_angles, q_robot_goal, threshold=0.3):
        dist = np.linalg.norm(np.array(q_robot_goal) - np.array(current_joint_angles))
        if dist <= threshold:
            return True
        else:
            return False
        
    def is_near_goal_W_space(self, world_to_eef, world_to_eef_goal, threshold=0.03):
        dist = np.linalg.norm(np.array(world_to_eef_goal[0]) - np.array(world_to_eef[0]))
        if dist <= threshold:
            return True
        else:
            return False
        
    def make_human_zero_mass(self):
        for j in range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)):
            self.bc.changeDynamics(self.humanoid._humanoid, j, mass=0.00001, physicsClientId=self.bc._client)
        
    def lock_human_joints(self, q_human):
        # Save original mass of each joint to restore later
        self.human_joint_masses = []
        for j in range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)):
            # Get the current dynamics info to save mass
            dynamics_info = self.bc.getDynamicsInfo(self.humanoid._humanoid, j, physicsClientId=self.bc._client)
            self.human_joint_masses.append(dynamics_info[0])  # Save mass (first item in tuple is mass)
            # Set mass to 0 to lock the joint
            self.bc.changeDynamics(self.humanoid._humanoid, j, mass=0, physicsClientId=self.bc._client)
        
        # Set arm joints velocities to 0
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, jointIndex=j, targetValue=q_human[i], targetVelocity=0, physicsClientId=self.bc._client)

    def lock_robot_arm_joints(self, robot, q_robot):
        # Save original mass of each joint to restore later
        self.robot_joint_masses = []
        for j in range(self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            dynamics_info = self.bc.getDynamicsInfo(robot.id, j, physicsClientId=self.bc._client)
            self.robot_joint_masses.append(dynamics_info[0])  # Save mass
            # Set mass to 0 to lock the joint
            self.bc.changeDynamics(robot.id, j, mass=0, physicsClientId=self.bc._client)
        
        # Set arm joints velocities to 0
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, jointIndex=joint_id, targetValue=q_robot[i], targetVelocity=0, physicsClientId=self.bc._client)
        
    def lock_robot_gripper_joints(self, robot):
        # Set arm joints velocities to 0
        for j in range(robot.eef_base_id, self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            self.bc.resetJointState(robot.id, jointIndex=j, targetValue=0, targetVelocity=0, physicsClientId=self.bc._client)

    def unlock_human_joints(self, q_human):
        # Restore the original mass for each joint to make them active
        for j in range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)):
            original_mass = self.human_joint_masses[j]
            self.bc.changeDynamics(self.humanoid._humanoid, j, mass=original_mass, physicsClientId=self.bc._client)
        
        # Restore the velocities
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, jointIndex=j, targetValue=q_human[i], physicsClientId=self.bc._client)

    def unlock_robot_arm_joints(self, robot, q_robot):
        # Restore the original mass for each joint to make them active
        for j in range(self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            original_mass = self.robot_joint_masses[j]
            self.bc.changeDynamics(robot.id, j, mass=original_mass, physicsClientId=self.bc._client)
        
        # Restore the velocities
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, jointIndex=joint_id, targetValue=q_robot[i], physicsClientId=self.bc._client)

    def generate_random_q_H(self):
        q_H_1 = np.random.uniform(-3.14, 3.14)
        q_H_2 = np.random.uniform(-0.24, 1.23)
        q_H_3 = np.random.uniform(-2.66, -1.32)
        q_H_4 = np.random.uniform(0.40, 2.54)
        return [q_H_1, q_H_2, q_H_3, q_H_4]
    
    def human_in_collision(self):
        """Check if any part of the human arm collides with other objects."""
        contact_points = self.bc.getContactPoints(bodyA=self.humanoid._humanoid, physicsClientId=self.bc._client)
        for point in contact_points:
            if (point[2] in [self.bed_id, self.cube_id, self.robot_2.id]):
                return True
        return False

    def reset_and_check(self):
        """Reset the human arm and check for collisions until no collision is detected."""
        while True:
            q_H = self.generate_random_q_H()
            self.reset_human_arm(q_H)
            self.bc.stepSimulation(physicsClientId=self.bc._client)
            if not self.human_in_collision():
                self.lock_human_joints(q_H)
                print(f'q_H: {q_H}')
                break

    def get_robot_joint_angles(self, robot):
        current_joint_angles = []
        for joint_id in robot.arm_controllable_joints:
            current_joint_angles.append(self.bc.getJointState(robot.id, joint_id)[0])
        return current_joint_angles

    def get_human_joint_angles(self):
        current_joint_angles = []
        for joint_id in self.human_controllable_joints:
            current_joint_angles.append(self.bc.getJointState(self.humanoid._humanoid, joint_id)[0])
        return current_joint_angles
    
    def interpolate_trajectory(self, robot_traj, alpha=0.5):
        new_traj = []
        for i in range(len(robot_traj) - 1):
            q_R_i = np.array(robot_traj[i])
            q_R_next = np.array(robot_traj[i + 1])
            
            interpolated_point = (1 - alpha) * q_R_i + alpha * q_R_next
            new_traj.append(robot_traj[i])  # Append the current point
            new_traj.append(interpolated_point.tolist())  # Append the interpolated point

        new_traj.append(robot_traj[-1])  # Append the last point to complete the trajectory

        return new_traj
    
    def get_init_traj_from_q_H(self, q_H_init, q_H_goal, q_R_init):
        q_H_traj = []
        q_H_traj.append(q_H_init)
        q_H_traj.append(q_H_goal)
        q_H_traj = self.interpolate_trajectory(q_H_traj, 0.5)
        q_H_traj = self.interpolate_trajectory(q_H_traj, 0.5)
        
        q_R_traj = []
        q_R_traj.append(q_R_init)
        prev_q_R = q_R_init
        
        for q_H in q_H_traj[1:]:
            self.reset_human_arm(q_H)
            q_R = self.get_q_R_from_elbow_pose(prev_q_R)
            q_R_traj.append(q_R)
            prev_q_R = q_R
    
        return q_H_traj, q_R_traj

    ### For debugging...
    def get_q_H_from_eef_pose(self):
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        world_to_cp = self.bc.multiplyTransforms(world_to_eef[0], world_to_eef[1],
                                                    self.eef_to_cp[0], self.eef_to_cp[1])
        world_to_right_elbow_joint = self.bc.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                                    self.cp_to_right_elbow_joint[0], self.cp_to_right_elbow_joint[1])
        q_H_goal = self.bc.calculateInverseKinematics(self.humanoid._humanoid, self.right_elbow, world_to_right_elbow_joint[0], world_to_right_elbow_joint[1],
                                                        self.human_arm_lower_limits, self.human_arm_upper_limits, 
                                                        self.human_arm_joint_ranges, self.human_rest_poses,
                                                        maxNumIterations=50)
        self.util.draw_frame(world_to_eef[0], world_to_eef[1])
        # self.util.draw_frame(world_to_cp[0], world_to_cp[1])
        # self.util.draw_frame(world_to_right_elbow_joint[0], world_to_right_elbow_joint[1])

        return q_H_goal
    
    def get_q_R_from_elbow_pose(self, prev_q_R):
        world_to_right_elbow_joint = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[4:6]
        world_to_cp = self.bc.multiplyTransforms(world_to_right_elbow_joint[0], world_to_right_elbow_joint[1],
                                                 self.right_elbow_joint_to_cp[0], self.right_elbow_joint_to_cp[1])
        world_to_eef = self.bc.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                                  self.cp_to_eef[0], self.cp_to_eef[1])
        q_R_goal = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                      self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, 
                                                      restPoses=prev_q_R,
                                                      maxNumIterations=50)
        q_R_goal = [q_R_goal[i] for i in range(len(self.robot.arm_controllable_joints))]
        # self.util.draw_frame(world_to_eef[0], world_to_eef[1])
        return q_R_goal
    
if __name__ == '__main__':
    manip_env = ManipulationDemo()
    manip_env.reset()

    bed_pcd = get_point_cloud_from_visual_shapes(manip_env.bed_id, client_id=manip_env.bc._client)
    manip_env.visualize_point_cloud(bed_pcd)
    print('here')