# ur5, pybullet
import os, inspect
import os.path as osp
import pybullet as p
import math
import sys

import pybullet_data
from pybullet_ur5.robot import UR5Robotiq85
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

# point cloud
import open3d as o3d
from utils.point_cloud_utils import *

# grasp generation
from utils.grasp_utils import *
from grasp_sampler.antipodal_grasp_sampler import *

# utils
from utils.collision_utils import get_collision_fn
from wiping_task.util import Util


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


class GraspDemo():
    def __init__(self, gui=False):
        # Start the bullet physics server
        self.gui = gui
        if self.gui:
            self.bc = BulletClient(connection_mode=p.GUI)
        else:
            self.bc = BulletClient(connection_mode=p.DIRECT)
        self.util = Util(self.bc._client)

    def reset(self):
        self.create_world()

    def create_world(self):
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.bc.setGravity(0, 0, -9.8) 
        self.bc.setGravity(0, 0, 0)
        self.bc.setTimestep = 0.05

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
        # self.robot_base_pose = ((0.65, 0.7, 0.25), (0, 0, -1.57))
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
        
        # initialize human parameters
        shoulder_min = [-3.141368118925281, -0.248997453133789, -2.6643015908664056]  # order: [yaw, pitch, roll]
        shoulder_max = [3.1415394736319917, 1.2392816988875348, -1.3229245882839409]  # order: [yaw, pitch, roll]
        elbow_min = [0.0]
        # elbow_min = [0.401146]
        elbow_max = [2.541304]
        self.human_arm_lower_limits = shoulder_min + elbow_min
        self.human_arm_upper_limits = shoulder_max + elbow_max

    def generate_grasps(self, q_H):
        def quaternion_dot(q1, q2):
            """ Compute the dot product of two quaternions """
            return np.dot(q1, q2)

        def check_perpendicularity(qA, qB):
            """ Check if the orientations are perpendicular """
            # Normalize quaternions to ensure correct dot product
            qA = qA / np.linalg.norm(qA)
            qB = qB / np.linalg.norm(qB)
            
            # Compute the dot product & compare deviation from zero
            dot_product = quaternion_dot(qA, qB)
            deviation = np.abs(dot_product)
            
            return deviation
        
        # initialize human arm and get its point cloud
        self.reset_human_arm(q_H)
        right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        m = R.from_quat(right_elbow[1]).as_matrix()
        inward_vec = m[:, 1]  # inward vec is the green axis (rgb axis)

        # generate object point cloud
        point_cloud = get_human_arm_pcd_for_grasp_sampler(self, client_id=self.bc._client)
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(point_cloud)
        pc_ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=15))
        pc_ply.orient_normals_consistent_tangent_plane(50)

        # generate antipodal grasp samples
        sampler = AntipodalGraspSampler(obj_inward_vector=inward_vec, max_num_surface_points=130, num_samples=7)
        prev_time = time.time()
        grasp_matrices = sampler.generate_grasps(pc_ply, vis=False)
        print(f"Generated {len(grasp_matrices)} grasps. Time: {time.time()-prev_time}.")

        # test each grasp sample
        q_R_grasp_samples = []
        grasp_pose_samples = []
        world_to_eef_goals = []
        best_q_R_grasp = None
        best_world_to_grasp = None
        best_world_to_eef_goal = None
        best_combined_score = float('inf')  # Initialize with a high value for comparison

        deviations = []
        distances = []

        for grasp in grasp_matrices:
            world_to_eef = self.bc.multiplyTransforms(grasp[:3, 3], quaternion_from_matrix(grasp),
                                                    self.eef_grasp_to_eef[0], self.eef_grasp_to_eef[1])
            q_R_grasp = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, 
                                                        world_to_eef[0], world_to_eef[1],
                                                        self.robot.arm_lower_limits, self.robot.arm_upper_limits, 
                                                        self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                        maxNumIterations=50)
            q_R_grasp = [q_R_grasp[i] for i in range(len(self.robot.arm_controllable_joints))]
            if q_R_grasp[-1] < -3.14:
                q_R_grasp[-1] += 3.14
            elif q_R_grasp[-1] > 3.14:
                q_R_grasp[-1] -= 3.14
            q_R_grasp = np.clip(q_R_grasp, self.robot.arm_lower_limits, self.robot.arm_upper_limits)

            self.reset_robot(self.robot, q_R_grasp)
            eef_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
            dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))

            if not self.robot_in_collision(q_R_grasp) and dist <= 0.02:
                q_R_grasp_samples.append(q_R_grasp)
                grasp_pose_samples.append([grasp[:3, 3], quaternion_from_matrix(grasp)])
                world_to_eef_goals.append(world_to_eef)

                # Calculate deviation from right elbow quaternion (Criteria 1)
                grasp_quaternion = quaternion_from_matrix(grasp)
                deviation = check_perpendicularity(right_elbow[1], grasp_quaternion)
                deviations.append(deviation)

                # Calculate distance from right elbow (Criteria 2)
                distance = np.linalg.norm(np.array(grasp[:3, 3]) - np.array(right_elbow[0]))
                distances.append(distance)

        # Normalize both the deviations and distances for scoring
        if deviations:
            deviations = np.array(deviations)
            distances = np.array(distances)

            deviations_norm = (deviations - deviations.min()) / (deviations.max() - deviations.min() + 1e-8)
            distances_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

            # Calculate weighted scores and select the best grasp
            for i in range(len(q_R_grasp_samples)):
                weighted_score = 0.4 * deviations_norm[i] + 0.6 * distances_norm[i]
                if weighted_score < best_combined_score:
                    best_combined_score = weighted_score
                    best_q_R_grasp = q_R_grasp_samples[i]
                    best_world_to_grasp = grasp_pose_samples[i]
                    best_world_to_eef_goal = world_to_eef_goals[i]

        print(f'No collision grasps: {len(q_R_grasp_samples)}')
        
        if len(q_R_grasp_samples) == 0:
            raise ValueError('No grasp available')

        return q_R_grasp_samples, grasp_pose_samples, best_q_R_grasp, best_world_to_grasp, best_world_to_eef_goal
    
    def set_grasp_parameters(self, right_elbow_to_cp, cp_to_right_elbow, eef_to_cp, cp_to_eef,
                             right_elbow_joint_to_cp, cp_to_right_elbow_joint, right_wrist_joint_to_cp, cp_to_right_wrist_joint):
        self.right_elbow_to_cp = right_elbow_to_cp
        self.cp_to_right_elbow = cp_to_right_elbow
        self.eef_to_cp = eef_to_cp
        self.cp_to_eef = cp_to_eef
        self.right_elbow_joint_to_cp = right_elbow_joint_to_cp 
        self.cp_to_right_elbow_joint = cp_to_right_elbow_joint
        self.right_wrist_joint_to_cp = right_wrist_joint_to_cp
        self.cp_to_right_wrist_joint = cp_to_right_wrist_joint

    def compute_grasp_parameters(self, q_H, q_R_grasp, grasp):
        # compute right_elbow_to_cp
        self.reset_human_arm(q_H)
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
        self.reset_robot(self.robot, q_R_grasp)
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1])
        eef_to_cp = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                               world_to_cp[0], world_to_cp[1])
        
        self.right_elbow_to_cp = right_elbow_to_cp
        self.T_right_elbow_joint_to_cp = T_right_elbow_joint_to_cp
        self.eef_to_cp = eef_to_cp

    def compute_q_R_goal(self, q_H, world_to_right_elbow):
        world_to_right_elbow = world_to_right_elbow
        world_to_cp = self.bc.multiplyTransforms(world_to_right_elbow[0], world_to_right_elbow[1],
                                                 self.right_elbow_to_cp[0], self.right_elbow_to_cp[1])
        cp_to_world = self.bc.invertTransform(world_to_cp[0], world_to_cp[1])
        eef_to_world = self.bc.multiplyTransforms(self.eef_to_cp[0], self.eef_to_cp[1],
                                                cp_to_world[0], cp_to_world[1])
        world_to_eef = self.bc.invertTransform(eef_to_world[0], eef_to_world[1])
        
        q_R_goal = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                    self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                    maxNumIterations=50)
        q_R_goal = [q_R_goal[i] for i in range(len(self.robot.arm_controllable_joints))]
        # if q_R_goal[-1] < -3.14:
        #     q_R_goal[-1] += 3.14
        # elif q_R_goal[-1] > 3.14:
        #     q_R_goal[-1] -= 3.14
        # q_R_goal = np.clip(q_R_goal, self.robot.arm_lower_limits, self.robot.arm_upper_limits)
        if min(q_R_goal) < min(self.robot.arm_lower_limits) or max(q_R_goal) > max(self.robot.arm_upper_limits):
            valid_grasp = False
            return valid_grasp, q_R_goal, world_to_eef
        
        self.reset_robot(self.robot, q_R_goal)
        self.robot_2.reset()
        self.reset_human_arm(q_H)

        # collision check
        if self.robot_in_collision(q_R_goal):
            valid_grasp = False
            return valid_grasp, q_R_goal, world_to_eef
        else:
            valid_grasp = True

        # reachability check
        eef_pos = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[0]
        dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pos))
        if dist > 0.03:
            valid_grasp = False
            return valid_grasp, q_R_goal, world_to_eef

        # distance to human check
        dist = self.get_human_to_robot_dist(q_H=q_H, q_robot=q_R_goal)
        if dist <= 0.05:
            valid_grasp = False
            return valid_grasp, q_R_goal, world_to_eef

        # distance check from robot to bed (exclude gripper fingers)
        dist = self.get_bed_to_robot_dist(q_robot=q_R_goal)
        if dist <= 0.03:
            valid_grasp = False
            return valid_grasp, q_R_goal, world_to_eef
        
        return valid_grasp, q_R_goal, world_to_eef
    
    def get_human_to_robot_dist(self, q_H, q_robot):
        self.reset_human_arm(q_H)
        self.reset_robot(self.robot, q_robot)
        self.bc.stepSimulation()
        min_dist = float('inf')
        for c in p.getClosestPoints(bodyA=self.robot.id, bodyB=self.humanoid._humanoid, distance=100, physicsClientId=self.bc._client):
            linkA = c[3]
            linkB = c[4]
            if linkB in self.human_right_arm:
                continue

            contact_distance = np.array(c[8])
            if contact_distance < min_dist:
                min_dist = contact_distance
                # print(f'linkA: {linkA}, linkB: {linkB}, dist: {contact_distance}')
        print(f'min dist human to robot: {min_dist}')
        return min_dist
    
    def get_bed_to_robot_dist(self, q_robot):
        self.reset_robot(self.robot, q_robot)
        self.bc.stepSimulation()
        min_dist = float('inf')
        for c in p.getClosestPoints(bodyA=self.robot.id, bodyB=self.bed_id, distance=100, physicsClientId=self.bc._client):
            linkA = c[3]
            linkB = c[4]
            if linkA >= 9:  # skip gripper fingers
                continue

            contact_distance = np.array(c[8])
            if contact_distance < min_dist:
                min_dist = contact_distance
                # print(f'linkA: {linkA}, linkB: {linkB}, dist: {contact_distance}')
        print(f'min dist bed to robot: {min_dist}')
        return min_dist

    def reset_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, joint_id, q_robot[i])

    def move_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_robot[i])

    def reset_human_arm(self, q_human):
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, j, q_human[i], physicsClientId=self.bc._client)

    def move_human_arm(self, q_human):
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.setJointMotorControl2(self.humanoid._humanoid, j, p.POSITION_CONTROL, q_human[i], physicsClientId=self.bc._client)

    def get_obstacle_point_cloud(self, obstacles):
        point_cloud = []
        for obstacle in obstacles:
            point_cloud.extend(get_point_cloud_from_visual_shapes(obstacle, client_id=self.bc._client))
        return np.array(point_cloud)


if __name__ == '__main__':
    env = GraspDemo(gui=True)
    env.reset()

    # initial joint states
    q_robot_init = env.robot.arm_rest_poses
    q_robot_2_init = env.robot_2.arm_rest_poses
    q_H_init = env.human_rest_poses

    # generate grasps
    q_R_grasp_samples, grasp_pose_samples, best_q_R_grasp, best_world_to_grasp, best_world_to_eef_goal = env.generate_grasps(q_H_init)
    env.compute_grasp_parameters(q_H_init, best_q_R_grasp, best_world_to_grasp)
    
    env.reset_robot(env.robot, best_q_R_grasp)
    print('done')