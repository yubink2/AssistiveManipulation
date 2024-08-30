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


class HumanDemo():
    def __init__(self):
        self.bc = BulletClient(connection_mode=p.GUI)
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc.setGravity(0, 0, -9.8) 
        self.bc.setTimestep = 0.05

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, 0, -0.04))
        self.bed_id = self.bc.loadURDF("./urdf/bed_0.urdf", (0.0, -0.1, 0.0), useFixedBase=True)  # bed
        self.cid = None

        # load human
        human_base_pos = (0, 0, 0.3)
        human_base_orn = self.bc.getQuaternionFromEuler((0, 1.57, 0))
        motionPath = 'deep_mimic/mocap/data/Sitting1.json'
        self.motion = MotionCaptureData()
        self.motion.Load(motionPath)
        self.humanoid = Humanoid(self.bc, self.motion, baseShift=human_base_pos, ornShift=human_base_orn)
        self.right_shoulder_y = 3
        self.right_shoulder_p = 4
        self.right_shoulder_r = 5
        self.right_shoulder = 6
        self.right_elbow = 7
        self.right_wrist = 8
        self.human_rest_poses = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]

        human_base = self.bc.getBasePositionAndOrientation(self.humanoid._humanoid)[:2]
        self.T_world_to_human_base = compute_matrix(translation=human_base[0], rotation=human_base[1])

        # initial and target human arm + q_H trajectory from h36m dataset
        self.q_H_traj = [[1.7, 0, -3.1, 0.2], [3.1, 0, -3., 0.3]]
        # self.q_H_traj = [[1.8, 1.2, -2.6, 0.45], [1.0, 0, -2.6, 1.8]]
        # self.q_H_traj = [[1.8, 0.5, -2.8, 0.2], [1.0, 0, -2.6, 1.8]]
        # self.q_H_traj = [[3.0, 0, -1.6, 0.45], [2.6, 1.2, -1.9, 1.4]]
        self.q_human_init = self.q_H_traj[0]
        self.q_human_goal = self.q_H_traj[-1]

        # load robot
        # self.robot_base_pose = ((0.65, 0.25, 0.25), (0, 0, -1.57))
        # cube_id = self.bc.loadURDF("./urdf/cube_0.urdf", 
        #                            (self.robot_base_pose[0][0], self.robot_base_pose[0][1], self.robot_base_pose[0][2]-0.15), useFixedBase=True)
        self.robot_base_pose = ((0.5, 0.7, 0), (0, 0, 0))
        self.world_to_robot_base = compute_matrix(translation=self.robot_base_pose[0], rotation=self.robot_base_pose[1], rotation_type='euler')
        self.robot = UR5Robotiq85(self.bc, self.robot_base_pose[0], self.robot_base_pose[1])
        self.robot.load()
        for _ in range(50):
            self.robot.reset()
            self.robot.open_gripper()

        # initialize robot parameters
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        world_to_eef_grasp = [
                [world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.14],
                world_to_eef[1]
            ]
        # draw_frame(self, world_to_eef_grasp[0], world_to_eef_grasp[1])
        eef_grasp_to_world = self.bc.invertTransform(world_to_eef_grasp[0], world_to_eef_grasp[1])
        eef_grasp_to_eef = self.bc.multiplyTransforms(eef_grasp_to_world[0], eef_grasp_to_world[1],
                                                           world_to_eef[0], world_to_eef[1])
        self.eef_grasp_to_eef = eef_grasp_to_eef

        # load second robot
        self.robot_2_base_pose = ((-0.5, 0.5, 0), (0, 0, 0))
        cube_id = p.loadURDF("./urdf/cube_0.urdf", 
                            (self.robot_2_base_pose[0][0], self.robot_2_base_pose[0][1], self.robot_2_base_pose[0][2]-0.15), useFixedBase=True,
                            physicsClientId=self.bc._client)
        self.robot_2 = UR5Robotiq85(self.bc, self.robot_2_base_pose[0], self.robot_2_base_pose[1])
        self.robot_2.load()
        self.robot_2.reset()

        # move second robot to sphere obstacle position
        q_robot_2 = self.bc.calculateInverseKinematics(self.robot_2.id, self.robot_2.eef_id, [0, 0.15, 0.85],
                                                    self.robot_2.arm_lower_limits, self.robot_2.arm_upper_limits, self.robot_2.arm_joint_ranges, self.robot_2.arm_rest_poses,
                                                    maxNumIterations=20)
        self.q_robot_2 = [q_robot_2[i] for i in range(len(self.robot_2.arm_controllable_joints))]

        for _ in range(10):
            self.reset_robot(self.robot_2, self.q_robot_2)
            self.bc.stepSimulation()

        # # initialize obstacles
        self.obstacles = []
        self.obstacles.append(self.bed_id)
        self.obstacles.append(self.robot_2.id)
        # self.obstacles.append(self.humanoid._humanoid)

        # get 'static' obstacle point cloud
        self.obs_pcd = self.get_obstacle_point_cloud(self.obstacles)

        # initialize collision checker        
        obstacles = self.obstacles
        obstacles.append(self.humanoid._humanoid)
        self.collision_fn = get_collision_fn(self.robot.id, self.robot.arm_controllable_joints, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

        # initialize robot config
        self.init_robot_configs()

    def generate_random_q_H(self):
        q_H_1 = np.random.uniform(1.8, 3.0)
        q_H_2 = np.random.uniform(-0.24, 1.23)
        q_H_3 = np.random.uniform(-2.66, -1.32)
        q_H_4 = np.random.uniform(0.40, 2.54)
        print(q_H_1, q_H_2, q_H_3, q_H_4)
        return [q_H_1, q_H_2, q_H_3, q_H_4]

    def generate_grasps(self, q_H):
        # init human arm and get its point cloud
        self.reset_human_arm(q_H)
        right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        m = R.from_quat(right_elbow[1]).as_matrix()
        inward_vec = m[:, 1]  # inward vec is the green axis (rgb axis)

        # generate object point cloud
        point_cloud = get_human_arm_pcd_for_grasp_sampler(self)
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(point_cloud)
        pc_ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=15))
        pc_ply.orient_normals_consistent_tangent_plane(50)
        # o3d.visualization.draw_geometries([pc_ply])

        # generate antipodal grasp samples
        sampler = AntipodalGraspSampler(obj_inward_vector=inward_vec, max_num_surface_points=200, num_samples=7)
        prev_time = time.time()
        grasp_matrices = sampler.generate_grasps(pc_ply, vis=False)
        print(f"Generated {len(grasp_matrices)} grasps. Time: {time.time()-prev_time}.")

        # test each grasp sample
        q_R_grasp_samples = []
        grasp_pose_samples = []
        for grasp in grasp_matrices:
            world_to_eef = self.bc.multiplyTransforms(grasp[:3, 3], quaternion_from_matrix(grasp),
                                                     self.eef_grasp_to_eef[0], self.eef_grasp_to_eef[1])
            q_R_grasp = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, 
                                                        world_to_eef[0], world_to_eef[1],
                                                        self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                        maxNumIterations=50)
            q_R_grasp = [q_R_grasp[i] for i in range(len(self.robot.arm_controllable_joints))]
            if q_R_grasp[-1] < -3.14:
                q_R_grasp[-1] += 3.14
            elif q_R_grasp[-1] > 3.14:
                q_R_grasp[-1] -= 3.14

            self.reset_robot(self.robot, q_R_grasp)
            eef_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
            dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))

            if not self.collision_fn(q_R_grasp) and dist <= 0.05:
                q_R_grasp_samples.append(q_R_grasp)
                grasp_pose_samples.append([grasp[:3, 3], quaternion_from_matrix(grasp)])

        print(f'no collision grasps: {len(q_R_grasp_samples)}')
        
        if len(q_R_grasp_samples) == 0:
            raise ValueError('no grasp available')

        return q_R_grasp_samples, grasp_pose_samples

    def init_robot_configs(self):
        # initial human arm and reset robot config
        self.robot.reset()
        self.reset_human_arm(self.q_human_init)
        self.bc.stepSimulation()

        # grasp generation
        # q_R_grasp_samples, grasp_pose_samples = self.generate_grasps(self.q_human_init)
        q_R_grasp_samples = [[-3.1261986468629, -1.8338786433187777, 1.4256020802013456, -0.9043742581934993, 0.004010565930590675, 1.3107086767560772]]
        grasp_pose_samples = [[[0.16082333042308455, 0.4625005289272614, 0.6342929578230712], [0.707388044876899, -0.7068249569936026, -0.0005628637715330792, 0.0005633121735972125]]]

        # collision check of the grasp for goal config
        count = -1
        min_dist_cp_idx = None
        min_dist_cp = math.inf
        final_q_R_grasp = []
        final_q_R_goal = []
        final_grasp_pose = []
        final_right_elbow_to_cp = []
        final_right_elbow_joint_to_cp = []
        final_eef_to_cp = []
        final_q_R_traj = []
        for q_R, grasp in zip(q_R_grasp_samples, grasp_pose_samples):
            # compute right_elbow_to_cp
            self.reset_human_arm(self.q_human_init)
            world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
            world_to_cp = (grasp[0], world_to_right_elbow[1])
            right_elbow_to_world = self.bc.invertTransform(world_to_right_elbow[0], world_to_right_elbow[1])
            right_elbow_to_cp = self.bc.multiplyTransforms(right_elbow_to_world[0], right_elbow_to_world[1],
                                                           world_to_cp[0], world_to_cp[1])

            # compute right_elbow_joint_to_cp
            world_to_right_elbow_joint = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[4:6]
            right_elbow_joint_to_world = self.bc.invertTransform(world_to_right_elbow_joint[0], world_to_right_elbow_joint[1])
            right_elbow_joint_to_cp = self.bc.multiplyTransforms(right_elbow_joint_to_world[0], right_elbow_joint_to_world[1],
                                                                 world_to_cp[0], world_to_cp[1])
            T_right_elbow_joint_to_cp = compute_matrix(translation=right_elbow_joint_to_cp[0], rotation=right_elbow_joint_to_cp[1])
            
            # compute eef_to_cp
            self.reset_robot(self.robot, q_R)
            world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)
            eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1])
            eef_to_cp = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                                   world_to_cp[0], world_to_cp[1])
            
            # flag
            valid_grasp = True
            q_R_traj = []

            for q_H in self.q_H_traj:
                # check grasp pose on each waypoint
                self.reset_human_arm(q_H)
                world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
                world_to_cp = self.bc.multiplyTransforms(world_to_right_elbow[0], world_to_right_elbow[1],
                                                        right_elbow_to_cp[0], right_elbow_to_cp[1])
                cp_to_world = self.bc.invertTransform(world_to_cp[0], world_to_cp[1])
                eef_to_world = self.bc.multiplyTransforms(eef_to_cp[0], eef_to_cp[1],
                                                        cp_to_world[0], cp_to_world[1])
                world_to_eef = self.bc.invertTransform(eef_to_world[0], eef_to_world[1])
                
                current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                            self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                            maxNumIterations=50)
                current_joint_angles = [current_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]
                if current_joint_angles[-1] < -3.14:
                    current_joint_angles[-1] += 3.14
                elif current_joint_angles[-1] > 3.14:
                    current_joint_angles[-1] -= 3.14
                self.reset_robot(self.robot, current_joint_angles)
                q_R_traj.append(current_joint_angles)

                # collision check
                eef_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
                dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))

                if self.collision_fn(current_joint_angles) or dist > 0.05:
                    valid_grasp = False
                    break

            # save satisfying states
            if valid_grasp:
                count += 1
                final_q_R_grasp.append(q_R)
                final_q_R_goal.append(current_joint_angles)
                final_grasp_pose.append(grasp)
                final_right_elbow_to_cp.append(right_elbow_to_cp)
                final_right_elbow_joint_to_cp.append(T_right_elbow_joint_to_cp)
                final_eef_to_cp.append(eef_to_cp)
                final_q_R_traj.append(q_R_traj)

                # dist_cp = np.linalg.norm(np.array(world_to_right_elbow[0]) - np.array(world_to_cp[0]))
                if dist < min_dist_cp:
                    min_dist_cp_idx = count

        # select final grasp
        print(f'no collision (init+waypoints+goal) grasps: {len(final_q_R_grasp)}')
        if len(final_q_R_grasp) == 0:
            raise ValueError('no grasp available')
        
        # save joint angles --> q_R_goal_before_grasp
        self.q_R_goal_before_grasp = final_q_R_grasp[min_dist_cp_idx]
        self.q_R_init_after_grasp = final_q_R_grasp[min_dist_cp_idx]
        self.q_R_goal_after_grasp = final_q_R_goal[min_dist_cp_idx]
        self.right_elbow_to_cp = final_right_elbow_to_cp[min_dist_cp_idx]
        self.T_right_elbow_joint_to_cp = final_right_elbow_joint_to_cp[min_dist_cp_idx]
        self.eef_to_cp = final_eef_to_cp[min_dist_cp_idx]
        self.q_R_traj = final_q_R_traj[min_dist_cp_idx]
        print('q_R_goal_before_grasp', self.q_R_goal_before_grasp)
        print('q_R_init_after_grasp', self.q_R_init_after_grasp)
        
    def update_pcd(self):
        link_to_separate = [self.right_elbow, self.right_wrist]
        human_pcd, separate_pcd = get_humanoid_point_cloud(self.humanoid._humanoid, link_to_separate)
        self.obs_pcd = np.vstack((self.obs_pcd, human_pcd))
        self.right_arm_pcd = np.array(separate_pcd)
    
        # update environment point cloud
        self.trajectory_planner.update_obstacle_pcd(self.obs_pcd)
        print("Updated environment point cloud")

    ### INITIALIZE PLANNER
    def init_traj_planner(self):
        JOINT_LIMITS = [
            np.array(self.robot.arm_lower_limits), 
            np.array(self.robot.arm_upper_limits)
        ]

        # order for humanoid urdf: [yaw, pitch, roll]
        shoulder_min = [-3.141368118925281, -0.248997453133789, -3.14] #-2.6643015908664056]
        shoulder_max = [3.1415394736319917, 1.2392816988875348, -1.3229245882839409]
        # shoulder_min = [-3.14, -3.14, -3.14]
        # shoulder_max = [3.14, 3.14, 3.14]
        elbow_min = [0]
        # elbow_min = [0.401146]
        elbow_max = [2.541304]
        self.human_arm_lower_limits = shoulder_min + elbow_min
        self.human_arm_upper_limits = shoulder_max + elbow_max

        # Instantiate mppi H clamp
        self.mppi_H_clamp = MPPI_H_Clamp(self.eef_to_cp, self.right_elbow_to_cp, self.robot_base_pose,
                                         self.human_arm_lower_limits, self.human_arm_upper_limits, human_rest_poses=self.q_human_goal)

        # Instantiate trajectory planner
        self.trajectory_planner = TrajectoryPlanner(
            joint_limits=JOINT_LIMITS,
            robot_urdf_location=robot_urdf_location,
            scene_urdf_location=scene_urdf_location,
            link_fixed=LINK_FIXED,
            link_ee=LINK_EE,
            link_skeleton=LINK_SKELETON,
            control_points_location = control_points_location,
            control_points_number = control_points_number,
            mppi_H_clamp = self.mppi_H_clamp,
            world_to_robot_base = self.world_to_robot_base,
        )
        print("Instantiated trajectory planner")

    def init_mppi_planner(self, current_joint_angles, target_joint_angles, clamp_by_human):
        # MPPI parameters
        N_JOINTS = len(self.robot.arm_controllable_joints)
        mppi_control_limits = [
            -0.05 * np.ones(N_JOINTS),
            0.05 * np.ones(N_JOINTS)
        ]
        mppi_nsamples = 500
        mppi_covariance = 0.005
        mppi_lambda = 1.0

        # Update current & target joint angles
        self.current_joint_angles = current_joint_angles
        self.target_joint_angles = target_joint_angles

        # Update whether to clamp_by_human
        self.clamp_by_human = clamp_by_human
        self.trajectory_planner.update_clamp_by_human(self.clamp_by_human)

        # Instantiate MPPI object
        self.trajectory_planner.instantiate_mppi_ja_to_ja(
            self.current_joint_angles,
            self.target_joint_angles,
            self.q_R_traj,
            mppi_control_limits=mppi_control_limits,
            mppi_nsamples=mppi_nsamples,
            mppi_covariance=mppi_covariance,
            mppi_lambda=mppi_lambda,
            waypoint_density = 5,
            action_smoothing= 0.4,
        )
        print('Instantiated MPPI object')

        # Update environment point cloud
        self.trajectory_planner.update_obstacle_pcd(self.obs_pcd)
        print("Updated environment point cloud")

    def get_mppi_trajectory(self, current_joint_angles):
        # Plan trajectory
        start_time = time.time()
        trajectory = self.trajectory_planner.get_mppi_rollout(current_joint_angles)
        print("planning time : ", time.time()-start_time)
        print(np.array(trajectory))
        return trajectory
    
    def init_traj_follower(self):
        JOINT_LIMITS = [
            np.array(self.robot.arm_lower_limits), 
            np.array(self.robot.arm_upper_limits)
        ]

        # Trajectory Follower initialization
        self.trajectory_follower = TrajectoryFollower(
            joint_limits = JOINT_LIMITS,
            robot_urdf_location = robot_urdf_location,
            control_points_json = control_points_location,
            link_fixed = LINK_FIXED,
            link_ee = LINK_EE,
            link_skeleton = LINK_SKELETON,
            control_points_number = control_points_number,
            world_to_robot_base = self.world_to_robot_base,
        )
        print('trajectory follower instantiated')

        # TODO update environment point cloud
        self.trajectory_follower.update_obstacle_pcd(self.obs_pcd)
        print("Updated environment point cloud")

    ### HELPER FUNCTIONS
    def attach_human_arm_to_eef(self, joint_type=p.JOINT_FIXED, attach_to_gripper=False):
        # attach human arm (obj) to eef (body)
        body_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]            # world to eef 
        obj_pose = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]    # world to cp
        world_to_body = self.bc.invertTransform(body_pose[0], body_pose[1])               # eef to world
        obj_to_body = self.bc.multiplyTransforms(world_to_body[0],                        # eef to cp
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])
        # self.eef_to_cp = obj_to_body

        if self.right_elbow_to_cp is None or self.eef_to_cp is None:
            raise ValueError('right_elbow_to_cp or eef_to_cp not initialized.')

        world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        world_to_cp = self.bc.multiplyTransforms(world_to_right_elbow[0], world_to_right_elbow[1],
                                                 self.right_elbow_to_cp[0], self.right_elbow_to_cp[1])
        
        
        self.cid = self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
                            parentLinkIndex=self.robot.eef_id,
                            childBodyUniqueId=self.humanoid._humanoid,
                            childLinkIndex=self.right_elbow,
                            jointType=joint_type,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=obj_to_body[0],
                            parentFrameOrientation=obj_to_body[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0))

        if attach_to_gripper:
            # compute transform matrix from robot's gripper to object frame
            T_eef_to_object = compute_matrix(translation=self.eef_to_cp[0], rotation=self.eef_to_cp[1], rotation_type='quaternion')

            # compute transform matrix for inverse of object pose in world frame
            T_world_to_object = compute_matrix(translation=world_to_cp[0], rotation=world_to_cp[1], rotation_type='quaternion')
            T_object_to_world = inverse_matrix(T_world_to_object)

            self.trajectory_planner.attach_to_gripper(object_type="pcd", object_geometry=self.right_arm_pcd,
                                                      T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world)

            return T_eef_to_object, T_object_to_world
    
    def deattach_human_arm_from_eef(self):
        if self.cid is not None:
            self.bc.removeConstraint(self.cid)
        self.cid = None

    def visualize_point_cloud(self, pcd):
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([pc_ply])

    def draw_sphere_marker(self, position, radius=0.07, color=[1, 0, 0, 1]):
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        col_id = self.bc.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        marker_id = self.bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vs_id)
        return marker_id 

    def human_motion_from_frame_data(self, humanoid, motion, utNum):
        keyFrameDuration = motion.KeyFrameDuraction()
        self.bc.stepSimulation()
        humanoid.RenderReference(utNum * keyFrameDuration, self.bc)
        self.bc.stepSimulation()

    def reset_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, joint_id, q_robot[i])

    def move_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_robot[i])

    def reset_human_arm(self, q_human):
        self.bc.resetJointState(self.humanoid._humanoid, self.right_shoulder_y, q_human[0])
        self.bc.resetJointState(self.humanoid._humanoid, self.right_shoulder_p, q_human[1])
        self.bc.resetJointState(self.humanoid._humanoid, self.right_shoulder_r, q_human[2])
        self.bc.resetJointState(self.humanoid._humanoid, self.right_elbow, q_human[3])

    def move_human_arm(self, q_human):
        self.bc.setJointMotorControl2(self.humanoid._humanoid, self.right_shoulder_y, p.POSITION_CONTROL, q_human[0])
        self.bc.setJointMotorControl2(self.humanoid._humanoid, self.right_shoulder_p, p.POSITION_CONTROL, q_human[1])
        self.bc.setJointMotorControl2(self.humanoid._humanoid, self.right_shoulder_r, p.POSITION_CONTROL, q_human[2])
        self.bc.setJointMotorControl2(self.humanoid._humanoid, self.right_elbow, p.POSITION_CONTROL, q_human[3])

    def get_obstacle_point_cloud(self, obstacles):
        point_cloud = []
        for obstacle in obstacles:
            if obstacle == self.humanoid._humanoid:
                # human_pcd, _ = get_humanoid_point_cloud(self.humanoid._humanoid, [self.right_elbow])
                # point_cloud.extend(human_pcd)
                continue
            elif obstacle == self.bed_id:
                half_extents = [0.5, 1.7, 0.2]
                point_cloud.extend(get_point_cloud_from_collision_shapes(obstacle, half_extents))
            else:
                point_cloud.extend(get_point_cloud_from_visual_shapes(obstacle))
        return np.array(point_cloud)
    
    def update_current_joint_angles(self, current_joint_angles):
        self.current_joint_angles = current_joint_angles

    def update_target_joint_angles(self, target_joint_angles):
        self.target_joint_angles = target_joint_angles

    def is_near_goal(self, current_joint_angles):
        dist = np.linalg.norm(np.array(self.q_R_goal_after_grasp) - np.array(current_joint_angles))
        if dist <= 0.75:
            return True
        else:
            return False

if __name__ == '__main__':
    env = HumanDemo()
    env.init_traj_planner()
    env.init_traj_follower()

    # ####

    # # Step 0: trajectory before grasping
    # traj = env.init_mppi_planner(env.q_R_init_before_grasp, env.q_R_goal_before_grasp, clamp_by_human=False)
    # print(traj)

    # for q in traj:
    #     for _ in range(100):
    #         env.move_robot(env.robot, q)
    #         env.move_human_arm(env.q_human_init)
    #         env.bc.stepSimulation()
    #     time.sleep(0.5)

    # for i, joint_id in enumerate(env.robot.arm_controllable_joints):
    #     print(i, env.bc.getJointState(env.robot.id, joint_id)[0])

    #####

    # TEST MOVING WITH HUMAN ARM ATTACHED
    # Step 1: move robot to grasping pose
    for _ in range(100):
        env.reset_robot(env.robot_2, env.q_robot_2)
        env.reset_robot(env.robot, env.q_R_goal_before_grasp)
        env.reset_human_arm(env.q_human_init)
        env.bc.stepSimulation()

    # Step 2: attach human arm to eef
    env.update_pcd()
    T_eef_to_object, T_object_to_world = env.attach_human_arm_to_eef(attach_to_gripper=True)

    # Step 3: trajectory after grasping
    env.init_mppi_planner(env.q_R_goal_before_grasp, env.q_R_goal_after_grasp, clamp_by_human=True)
    traj = env.get_mppi_trajectory(env.q_R_goal_before_grasp)
    previous_update_time = time.time()
    update_second = 20  # sec

    # Step 4: initialize trajectory follower
    env.trajectory_follower.update_obstacle_pcd(env.obs_pcd)
    env.trajectory_follower.update_trajectory(traj)
    env.trajectory_follower.attach_to_gripper(object_type="pcd", object_geometry=env.right_arm_pcd,
                                              T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world,
                                              T_world_to_human_base=env.T_world_to_human_base, T_right_elbow_joint_to_cp=env.T_right_elbow_joint_to_cp,
                                              human_arm_lower_limits=env.human_arm_lower_limits, human_arm_upper_limits=env.human_arm_upper_limits)
    current_joint_angles = env.q_R_goal_before_grasp
    current_human_joint_angles = env.q_human_init

    # env.deattach_human_arm_from_eef()
    # env.reset_human_arm(env.q_human_goal)

    # Step 5: simulation loop
    while True:
        # if near goal, execute rest of trajectory and end simulation loop
        if env.is_near_goal(current_joint_angles):
            for q in traj:
                for _ in range (100):
                    env.reset_robot(env.robot_2, env.q_robot_2)
                    env.move_robot(env.robot, q)
                    env.bc.stepSimulation()
                time.sleep(1.0)
            break

        # get velocity command
        prev_time = time.time()
        velocity_command = env.trajectory_follower.follow_trajectory(current_joint_angles, current_human_joint_angles)[0]
        current_time = time.time()
        print('following time: ', current_time-prev_time)

        # update trajectory 
        if current_time-previous_update_time > update_second:
            print('replanning...')
            traj = env.get_mppi_trajectory(current_joint_angles)
            env.trajectory_follower.update_trajectory(traj)
            previous_update_time = time.time()

        # if valid velocity command, move robot
        else:
            joint_angles = current_joint_angles + velocity_command * 0.2
            for _ in range (100):
                env.reset_robot(env.robot_2, env.q_robot_2)
                env.move_robot(env.robot, joint_angles)
                env.bc.stepSimulation()

            # save current_joint_angle
            current_joint_angles = []
            for joint_id in env.robot.arm_controllable_joints:
                current_joint_angles.append(env.bc.getJointState(env.robot.id, joint_id)[0])
            current_human_joint_angles = []
            current_human_joint_angles.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_y)[0])
            current_human_joint_angles.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_p)[0])
            current_human_joint_angles.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_r)[0])
            current_human_joint_angles.append(env.bc.getJointState(env.humanoid._humanoid, env.right_elbow)[0])

    print('done')
    ###

    # ## TEST MOVING WITHOUT HUMAN ARM ATTACHED

    # # Step 1: move robot to grasping pose
    # for _ in range(100):
    #     env.reset_robot(env.robot_2, env.q_robot_2)
    #     env.reset_robot(env.robot, env.q_R_goal_before_grasp)
    #     env.bc.stepSimulation()

    # # Step 2: initialize trajectory planner and get mppi trajectory (after grasping)
    # env.init_mppi_planner(env.q_R_goal_before_grasp, env.q_R_goal_after_grasp, clamp_by_human=False)
    # traj = env.get_mppi_trajectory(env.q_R_goal_before_grasp)
    # previous_update_time = time.time()
    # update_second = 5  # sec

    # # Step 3: initialize trajectory follower
    # env.init_traj_follower()
    # env.trajectory_follower.update_trajectory(traj)
    # current_joint_angles = env.q_R_goal_before_grasp

    # # Step 4: simulation loop
    # while True:
    #     # get velocity command
    #     prev_time = time.time()
    #     velocity_command = env.trajectory_follower.follow_trajectory(current_joint_angles)[0]
    #     current_time = time.time()
    #     print('following time: ', current_time-prev_time)

    #     # update trajectory 
    #     if current_time-previous_update_time > update_second:
    #         traj = env.get_mppi_trajectory(current_joint_angles)
    #         env.trajectory_follower.update_trajectory(traj)
    #         previous_update_time = time.time()

    #     # if valid velocity command, move robot
    #     else:
    #         for i, joint_id in enumerate(env.robot.arm_controllable_joints):
    #             env.bc.setJointMotorControl2(
    #                     env.robot.id, joint_id,
    #                     controlMode = p.VELOCITY_CONTROL,
    #                     targetVelocity = velocity_command[i]
    #                 )
    #         env.reset_robot(env.robot_2, env.q_robot_2)
    #         env.bc.stepSimulation()

    #         # save current_joint_angle
    #         current_joint_angles = []
    #         for joint_id in env.robot.arm_controllable_joints:
    #             current_joint_angles.append(env.bc.getJointState(env.robot.id, joint_id)[0])

    # #####
