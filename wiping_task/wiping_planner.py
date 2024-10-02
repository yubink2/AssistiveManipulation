import os, inspect
import numpy as np
import time
import sys

import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

# humanoid
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from deep_mimic.env.motion_capture_data import MotionCaptureData
from deep_mimic.mocap.humanoid_with_rev_xyz import Humanoid

# robot
from pybullet_ur5.robot import UR5Robotiq85
from utils.collision_utils import get_collision_fn

# utils
from wiping_task.util import Util
from wiping_task.targets_util import TargetsUtil
from wiping_task.score_util import ScoreUtil
from utils.transform_utils import *

class WipingDemo():
    def __init__(self, gui=False):
        # Start the bullet physics server
        self.gui = gui
        if self.gui:
            self.bc = BulletClient(connection_mode=p.GUI)
        else:
            self.bc = BulletClient(connection_mode=p.DIRECT)
        self.util = Util(self.bc._client)
        self.targets_util = TargetsUtil(self.bc._client, self.util)
        self.score_util = ScoreUtil(self.bc._client, self.util)

    def reset(self):
        self.create_world()
        self.init_tool()
        self.targets_util.init_targets_util(self.humanoid._humanoid, self.right_shoulder, self.right_elbow, self.human_right_arm,
                                            self.robot_2, self.tool,
                                            self.target_to_eef, self.target_closer_to_eef, self.robot_2_in_collision)
        self.score_util.init_score_util(self.humanoid._humanoid, self.right_shoulder, self.right_elbow, self.human_controllable_joints,
                                        self.robot, self.robot_2, self.tool,
                                        self.target_to_eef, self.target_closer_to_eef, self.robot_2_in_collision, self.robot_in_collision)
        
        self.targets_util.generate_new_targets_pose()
        self.targets_util.generate_targets()
        self.targets_util.initialize_deleted_targets_list()
        
    def reset_wiping_setup(self, q_H, targeted_arm):
        self.reset_human_arm(q_H)
        self.lock_human_joints(q_H)
        self.targets_util.update_targets()

        # feasible targets
        feasible_targets_found = self.targets_util.get_feasible_targets_pos(targeted_arm=targeted_arm)
        if not feasible_targets_found:
            return feasible_targets_found
        
        self.targets_util.reorder_feasible_targets(targeted_arm=targeted_arm)
        self.targets_util.mark_feasible_targets()
        return feasible_targets_found

    def create_world(self):
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc.setGravity(0, 0, 0, physicsClientId=self.bc._client) 

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, 0, -0.04), physicsClientId=self.bc._client)
        self.bed_id = self.bc.loadURDF("./urdf/bed_0.urdf", (0.0, -0.1, 0.0), useFixedBase=True)

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

        # load first robot (manipulation)
        self.robot_base_pose = ((0.5, 0.8, 0.25), (0, 0, 0))
        self.cube_id = self.bc.loadURDF("./urdf/cube_0.urdf", 
                                   (self.robot_base_pose[0][0], self.robot_base_pose[0][1], self.robot_base_pose[0][2]-0.15), useFixedBase=True)
        self.robot = UR5Robotiq85(self.bc, self.robot_base_pose[0], self.robot_base_pose[1])
        self.robot.load()
        for _ in range(50):
            self.robot.reset()
            self.robot.open_gripper()

        # load second robot (wiping)
        self.robot_2_base_pose = ((0.55, 0, 0), (0, 0, -1.57))
        self.cube_2_id = p.loadURDF("./urdf/cube_0.urdf", 
                            (self.robot_2_base_pose[0][0], self.robot_2_base_pose[0][1], self.robot_2_base_pose[0][2]-0.15), useFixedBase=True,
                            physicsClientId=self.bc._client)
        self.robot_2 = UR5Robotiq85(self.bc, self.robot_2_base_pose[0], self.robot_2_base_pose[1])
        self.robot_2.load()
        self.robot_2.reset()

        self.targets_pos_on_upperarm = None
        self.targets_pos_on_forearm = None

        # initialize collision checker (robot 2)
        obstacles = [self.bed_id, self.humanoid._humanoid, self.robot.id, self.cube_id]
        self.robot_2_in_collision = get_collision_fn(self.robot_2.id, self.robot_2.arm_controllable_joints, obstacles=obstacles,
                                                     attachments=[], self_collisions=True,
                                                     disabled_collisions=set(), client_id=self.bc._client)
        
        # initialize collision checker (robot)
        robot_obstacles = [self.bed_id, self.robot_2.id, self.cube_2_id, self.humanoid._humanoid]
        self.robot_in_collision = get_collision_fn(self.robot.id, self.robot.arm_controllable_joints, obstacles=robot_obstacles,
                                                   attachments=[], self_collisions=True,
                                                   disabled_collisions=set(), client_id=self.bc._client)
        
        # compute target_to_eef & target_closer_to_eef
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        target_orn = self.util.rotate_quaternion_by_axis(world_to_eef[1], axis='z', degrees=180)
        world_to_target = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.3], target_orn]
        world_to_target_closer = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.189], target_orn]

        self.target_orn = target_orn
        target_to_world = self.bc.invertTransform(world_to_target[0], world_to_target[1], physicsClientId=self.bc._client)
        target_closer_to_world = self.bc.invertTransform(world_to_target_closer[0], world_to_target_closer[1], physicsClientId=self.bc._client)
        self.target_to_eef = self.bc.multiplyTransforms(target_to_world[0], target_to_world[1],
                                                        world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        self.target_closer_to_eef = self.bc.multiplyTransforms(target_closer_to_world[0], target_closer_to_world[1],
                                                               world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
    
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

        if not self.gui:
            self.bc.resetBasePositionAndOrientation(self.tool, [100,100,100], [0,0,0,1], physicsClientId=self.bc._client)
        
    def attach_tool(self):
        # reset tool and attach it to eef
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        world_to_tool = self.bc.multiplyTransforms(world_to_eef[0], world_to_eef[1],
                                                self.eef_to_tool[0], self.eef_to_tool[1], physicsClientId=self.bc._client)
        self.bc.resetBasePositionAndOrientation(self.tool, world_to_tool[0], world_to_tool[1], physicsClientId=self.bc._client)

        # create constraint that keeps the tool in the gripper
        self.cid = self.bc.createConstraint(parentBodyUniqueId=self.robot_2.id,
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
        if self.cid is not None:
            self.bc.removeConstraint(self.cid)
        self.cid = None
        self.bc.resetBasePositionAndOrientation(self.tool, [100,100,100], [0,0,0,1], physicsClientId=self.bc._client)

    def compute_feasible_targets_robot_traj(self):
        robot_traj = []
        prev_target_pos_world = None

        for target_pos_world, target_orn_world in zip(self.targets_util.feasible_targets_pos_world, self.targets_util.feasible_targets_orn_world):
            # # check for invalid order of target sequence
            # if prev_target_pos_world is not None:
            #     dist = np.linalg.norm(np.array(prev_target_pos_world) - np.array(target_pos_world))
            #     if dist > 0.035:
            #         break

            # compute desired world_to_eef (check if it can get closer to the target point)
            world_to_eef = self.bc.multiplyTransforms(target_pos_world, target_orn_world,
                                                      self.target_closer_to_eef[0], self.target_closer_to_eef[1], physicsClientId=self.bc._client)

            # set robot initial joint state
            q_robot_2_closer = self.bc.calculateInverseKinematics(self.robot_2.id, self.robot_2.eef_id, world_to_eef[0], world_to_eef[1],
                                                            lowerLimits=self.robot_2.arm_lower_limits, upperLimits=self.robot_2.arm_upper_limits, 
                                                            jointRanges=self.robot_2.arm_joint_ranges, restPoses=self.robot_2.arm_rest_poses,
                                                            maxNumIterations=50, physicsClientId=self.bc._client)
            q_robot_2_closer = [q_robot_2_closer[i] for i in range(len(self.robot_2.arm_controllable_joints))]
            q_robot_2_closer = np.clip(q_robot_2_closer, self.robot_2.arm_lower_limits, self.robot_2.arm_upper_limits)

            for i, joint_id in enumerate(self.robot_2.arm_controllable_joints):
                self.bc.resetJointState(self.robot_2.id, joint_id, q_robot_2_closer[i], physicsClientId=self.bc._client)
            self.bc.stepSimulation(physicsClientId=self.bc._client)

            # check if config is valid
            eef_pose = p.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
            pos_dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))
            dot_product = np.abs(np.dot(world_to_eef[1], eef_pose[1]))
            orn_dist = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
            if not self.robot_2_in_collision(q_robot_2_closer) and pos_dist <= 0.02 and orn_dist < np.deg2rad(15):
                robot_traj.append(q_robot_2_closer)

            prev_target_pos_world = target_pos_world

        self.robot_2.reset()

        return robot_traj
    
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
    
    def reset_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, joint_id, q_robot[i], physicsClientId=self.bc._client)

    def move_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_robot[i], physicsClientId=self.bc._client)

    def reset_human_arm(self, q_human):
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, j, q_human[i], physicsClientId=self.bc._client)

    def generate_random_q_H(self):
        q_H_1 = np.random.uniform(0, 3.14)
        q_H_2 = np.random.uniform(-0.24, 1.23)
        q_H_3 = np.random.uniform(-2.66, -1.32)
        q_H_4 = np.random.uniform(0.40, 2.54)
        return [q_H_1, q_H_2, q_H_3, q_H_4] 
    
    def human_in_collision(self):
        """Check if any part of the human arm collides with other objects."""
        contact_points = self.bc.getContactPoints(bodyA=self.humanoid._humanoid, physicsClientId=self.bc._client)
        for point in contact_points:
            if (point[2] in [self.bed_id]):
                return True
        return False

    def get_valid_q_H(self):
        """Reset the human arm and check for collisions until no collision is detected."""
        for _ in range(5000):
            q_H = self.generate_random_q_H()
            self.reset_human_arm(q_H)
            self.bc.stepSimulation(physicsClientId=self.bc._client)
            if not self.human_in_collision():
                self.lock_human_joints(q_H)
                world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
                return q_H, world_to_right_elbow
        raise ValueError('valid human config not found!')

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
            self.bc.resetJointState(self.robot.id, jointIndex=joint_id, targetValue=q_robot[i], targetVelocity=0, physicsClientId=self.bc._client)

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
            self.bc.resetJointState(self.robot.id, jointIndex=joint_id, targetValue=q_robot[i], physicsClientId=self.bc._client)

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
    
    def get_eef_pose(self, robot, current_joint_angles, target_joint_angles):
        self.reset_robot(robot, target_joint_angles)
        eef_pose = self.bc.getLinkState(robot.id, robot.eef_id)[:2]
        self.reset_robot(robot, current_joint_angles)
        return eef_pose
    
    def step(self):
        self.bc.stepSimulation(physicsClientId=self.bc._client)
        self.targets_util.update_targets()

    def get_score(self, q_H_init, q_H_goal, q_robot):
        self.targets_util.update_targets()
        self.score_util.reset(targets_pos_upperarm_world=self.targets_util.targets_pos_upperarm_world, 
                              targets_orn_upperarm_world=self.targets_util.targets_orn_upperarm_world, 
                              targets_pos_forearm_world=self.targets_util.targets_pos_forearm_world, 
                              targets_orn_forearm_world=self.targets_util.targets_orn_forearm_world, 
                              q_H=q_H_goal, q_robot=q_robot)

        feasibility_score = self.score_util.compute_score_by_feasibility()
        # closeness_score = self.score_util.compute_score_by_closeness(q_H_init, q_H_goal)
        # total_score = 0.9*feasibility_score + 0.1*closeness_score

        total_score = feasibility_score

        return total_score
    
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
        # print(f'min dist human to robot: {min_dist}')
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
        # print(f'min dist bed to robot: {min_dist}')
        return min_dist

    def validate_q_R_goal(self, q_H, world_to_right_elbow, q_R_goal):
        world_to_right_elbow = world_to_right_elbow
        world_to_cp = self.bc.multiplyTransforms(world_to_right_elbow[0], world_to_right_elbow[1],
                                                 self.right_elbow_to_cp[0], self.right_elbow_to_cp[1])
        cp_to_world = self.bc.invertTransform(world_to_cp[0], world_to_cp[1])
        eef_to_world = self.bc.multiplyTransforms(self.eef_to_cp[0], self.eef_to_cp[1],
                                                cp_to_world[0], cp_to_world[1])
        world_to_eef = self.bc.invertTransform(eef_to_world[0], eef_to_world[1])

        valid_grasp = True

        # feasibility check
        if min(q_R_goal) < min(self.robot.arm_lower_limits) or max(q_R_goal) > max(self.robot.arm_upper_limits):
            valid_grasp = False
            return valid_grasp
        
        self.reset_robot(self.robot, q_R_goal)
        self.robot_2.reset()
        self.reset_human_arm(q_H)

        # collision check
        if self.robot_in_collision(q_R_goal):
            valid_grasp = False
            return valid_grasp

        # reachability check
        eef_pos = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[0]
        dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pos))
        if dist > 0.03:
            valid_grasp = False
            return valid_grasp

        # distance to human check
        dist = self.get_human_to_robot_dist(q_H=q_H, q_robot=q_R_goal)
        if dist <= 0.05:
            valid_grasp = False
            return valid_grasp

        # distance check from robot to bed (exclude gripper fingers)
        dist = self.get_bed_to_robot_dist(q_robot=q_R_goal)
        if dist <= 0.03:
            valid_grasp = False
            return valid_grasp
        
        return valid_grasp
    
    def validate_q_R(self, q_H, q_R, check_goal=False):
        # feasibility check
        if min(q_R) < min(self.robot.arm_lower_limits) or max(q_R) > max(self.robot.arm_upper_limits):
            return False
        
        self.reset_robot(self.robot, q_R)
        self.robot_2.reset()
        self.reset_human_arm(q_H)
        
        world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        world_to_cp = self.bc.multiplyTransforms(world_to_right_elbow[0], world_to_right_elbow[1],
                                                 self.right_elbow_to_cp[0], self.right_elbow_to_cp[1])
        cp_to_world = self.bc.invertTransform(world_to_cp[0], world_to_cp[1])
        eef_to_world = self.bc.multiplyTransforms(self.eef_to_cp[0], self.eef_to_cp[1],
                                                cp_to_world[0], cp_to_world[1])
        world_to_eef = self.bc.invertTransform(eef_to_world[0], eef_to_world[1])
        
        # collision check
        if self.robot_in_collision(q_R):
            return False

        # reachability check
        eef_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pose[0]))
        if dist > 0.03:
            return False

        if check_goal:
            # distance to human check
            dist = self.get_human_to_robot_dist(q_H=q_H, q_robot=q_R)
            if dist <= 0.05:
                return False

            # distance check from robot to bed (exclude gripper fingers)
            dist = self.get_bed_to_robot_dist(q_robot=q_R)
            if dist <= 0.03:
                return False
        
        return True

    def get_init_traj_from_q_H(self, q_H_init, q_H_goal, q_R_init):
        q_H_traj = []
        q_H_traj.append(q_H_init)
        q_H_traj.append(q_H_goal)
        q_H_traj = self.interpolate_trajectory(q_H_traj, 0.5)
        q_H_traj = self.interpolate_trajectory(q_H_traj, 0.5)
        
        q_R_traj = []
        prev_q_R = q_R_init
        
        for q_H in q_H_traj:
            self.reset_human_arm(q_H)
            q_R = self.get_q_R_from_elbow_pose(prev_q_R)
            q_R_traj.append(q_R)
            prev_q_R = q_R
    
        return q_H_traj, q_R_traj

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
        return q_R_goal
    
    def get_best_valid_goal_configs(self, q_H_init, q_robot, q_robot_2):
        q_H_trajs = []
        q_R_trajs = []
        q_H_goals = []
        q_R_goals = []
        count = 0
        while True:
            valid_grasp = False
            if count >= 30:
                break
            self.reset_robot(self.robot_2, q_robot_2)
            q_H_goal, world_to_right_elbow = self.get_valid_q_H()
            q_H_traj, q_R_traj = self.get_init_traj_from_q_H(q_H_init=q_H_init, q_H_goal=q_H_goal, q_R_init=q_robot)
            q_R_goal = q_R_traj[-1]
            valid_grasp = self.validate_q_R_goal(q_H_goal, world_to_right_elbow, q_R_goal)
            if valid_grasp:
                q_H_trajs.append(q_H_traj)
                q_R_trajs.append(q_R_traj)
                q_H_goals.append(q_H_goal)
                q_R_goals.append(q_R_goal)
                count += 1

        q_H_scores = []
        for q_H_goal, q_R_goal in zip(q_H_goals, q_R_goals):
            self.lock_human_joints(q_H_goal)
            self.lock_robot_arm_joints(self.robot, q_R_goal)
            score = self.get_score(q_H_init=q_H_init, q_H_goal=q_H_goal, q_robot=q_R_goal)
            q_H_scores.append(score)

        # sort by scores
        combined = zip(q_H_scores, q_H_trajs, q_R_trajs, q_H_goals, q_R_goals)  # zip
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)  # in descending order
        q_H_scores, q_H_trajs, q_R_trajs, q_H_goals, q_R_goals = zip(*sorted_combined)  # unzip

        q_H_scores = list(q_H_scores)
        q_H_trajs = list(q_H_trajs)
        q_R_trajs = list(q_R_trajs)
        q_H_goals = list(q_H_goals)
        q_R_goals = list(q_R_goals)

        # validate each waypoint
        for idx, (q_H_traj, q_R_traj) in enumerate(zip(q_H_trajs, q_R_trajs)):
            is_valid = True
            for q_H, q_R in zip(q_H_traj, q_R_traj):
                self.reset_human_arm(q_H)
                self.reset_robot(self.robot, q_R)
                is_valid = is_valid and self.validate_q_R(q_H, q_R)
            if is_valid:
                print(idx)
                break

        # reset environment
        self.lock_human_joints(q_H_init)
        self.lock_robot_arm_joints(self.robot, q_robot)
        self.reset_robot(self.robot_2, q_robot_2)

        return q_H_scores[idx], q_H_trajs[idx], q_R_trajs[idx], q_H_goals[idx], q_R_goals[idx]


if __name__ == '__main__':

    env = WipingDemo(gui=True)
    env.reset()

    q_robot_init = env.robot.arm_rest_poses
    q_robot_2_init = env.robot_2.arm_rest_poses
    q_H_init = env.human_rest_poses
    env.lock_human_joints(q_H_init)

    ### 2: test feasible targets selection
    start_time = time.time()
    arms = ['upperarm', 'forearm']
    total_targets_cleared = 0
    total_targets = env.targets_util.total_target_count
    success_rate = 0.0
    q_H = q_H_init
    for arm in arms:
        for _ in range(3):
            env.lock_robot_arm_joints(env.robot, q_robot_init)
            env.reset_robot(env.robot_2, q_robot_2_init)

            feasible_targets_found = env.reset_wiping_setup(q_H, arm)
            if not feasible_targets_found:
                print(f'{q_H}, {arm}, feasible targets not found!')
                continue
            
            robot_traj = env.compute_feasible_targets_robot_traj()
            if len(robot_traj) == 0:
                print(f'{q_H}, {arm}, valid trajectory not found!')
                continue
            robot_traj = env.interpolate_trajectory(robot_traj, alpha=0.5)
            robot_traj = env.interpolate_trajectory(robot_traj, alpha=0.25)
            env.reset_robot(env.robot_2, q_robot_2_init)

            # reset to initial config
            for _ in range(50):
                env.reset_robot(env.robot_2, env.targets_util.init_q_R)
            env.attach_tool()

            # execute wiping trajectory
            targets_cleared = 0
            for q_R in robot_traj:
                env.move_robot(env.robot_2, q_R)
                for _ in range(50):
                    env.bc.stepSimulation()
                new_target, indices_to_delete = env.targets_util.get_new_contact_points(targeted_arm=arm)
                env.targets_util.remove_contacted_feasible_targets(indices_to_delete, arm)
                targets_cleared += new_target
                total_targets_cleared += new_target
            print(f'targets_cleared: {targets_cleared}, total_targets_cleared: {total_targets_cleared}')
            print(f'wiping {arm} is done')

            env.targets_util.remove_targets()
            env.targets_util.unmark_feasible_targets()
            env.targets_util.update_targets()
            env.detach_tool()
    env.reset_robot(env.robot_2, q_robot_2_init)
    env.bc.stepSimulation()

    ## 1: test scoring function
    best_q_R_grasp = [-0.9476715,  -1.67301685,  1.98052876, -1.50240573, -1.23648364,  0.48277612]
    best_world_to_grasp = [[0.47961152, 0.35232658, 0.41274854], [ 0.85635702, -0.00354509, -0.51330226,  0.0562217 ]]
    best_world_to_eef_goal = ((0.4130373836953966, 0.3616619928431144, 0.5361486636730375), (0.856393844304345, -0.0034889788426920583, -0.5132398108120827, 0.05623439394034839))
    env.compute_grasp_parameters(q_H_init, best_q_R_grasp, best_world_to_grasp)


    for i in range(50):
        env.bc.configureDebugVisualizer(flag=p.COV_ENABLE_RENDERING, enable=0)
        # q_H_goals, world_to_right_elbows, q_H_scores, q_R_goals, world_to_eef_goals = env.get_best_valid_q_H(q_H_init=q_H_init, q_robot=q_robot_init, q_robot_2=q_robot_2_init)
        q_H_score, q_H_traj, q_R_traj, q_H_goal, q_R_goal = env.get_best_valid_goal_configs(q_H_init=q_H_init, 
                                                                                            q_robot=q_robot_init, 
                                                                                            q_robot_2=q_robot_2_init)
        env.bc.configureDebugVisualizer(flag=p.COV_ENABLE_RENDERING, enable=1)
        env.lock_human_joints(q_H_goal)
        env.reset_robot(env.robot, q_R_goal)
        q_H = q_H_goal
        print('q_H score: ', q_H_score)
        
        ### 2: test feasible targets selection
        arms = ['upperarm', 'forearm']
        for arm in arms:
            for _ in range (3):
                # env.lock_robot_arm_joints(env.robot, q_R_goal)
                feasible_targets_found = env.reset_wiping_setup(q_H, arm)
                env.reset_robot(env.robot_2, q_robot_2_init)
                if not feasible_targets_found:
                    # print(f'{q_H}, {arm}, feasible targets not found!')
                    continue
                
                robot_traj = env.compute_feasible_targets_robot_traj()
                env.reset_robot(env.robot_2, q_robot_2_init)
                if len(robot_traj) == 0:
                    # print(f'{q_H}, {arm}, valid trajectory not found!') 
                    continue
                robot_traj = env.interpolate_trajectory(robot_traj, alpha=0.5)
                robot_traj = env.interpolate_trajectory(robot_traj, alpha=0.25)

                # reset to initial config
                for _ in range(50):
                    env.reset_robot(env.robot_2, env.targets_util.init_q_R)
                env.attach_tool()

                # execute wiping trajectory
                targets_cleared = 0
                for q_R in robot_traj:
                    for _ in range(100):
                        env.move_robot(env.robot_2, q_R)
                        env.bc.stepSimulation()
                    new_target, indices_to_delete = env.targets_util.get_new_contact_points(targeted_arm=arm)
                    env.targets_util.remove_contacted_feasible_targets(indices_to_delete, arm)  
                    targets_cleared += new_target
                    total_targets_cleared += new_target
                success_rate = total_targets_cleared/total_targets
                print(f'targets_cleared: {targets_cleared}, total_targets_cleared: {total_targets_cleared}/{total_targets}')
                print(f'wiping {arm} is done')

                env.targets_util.remove_targets()
                env.targets_util.unmark_feasible_targets()  
                env.targets_util.update_targets()
                env.detach_tool()
                env.reset_robot(env.robot_2, q_robot_2_init)

        env.reset_robot(env.robot_2, q_robot_2_init)
        if success_rate >= 0.8:
            break

    total_time = time.time() - start_time
    print('here')
    print(f'total_targets_cleared: {total_targets_cleared}/{total_targets}')
    print(f'iter: {i}, total_time: {total_time}')