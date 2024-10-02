# ur5, pybullet
import os, inspect
import os.path as osp
import pybullet as p
import math
import sys
sys.path.append("/usr/lib/python3/dist-packages")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import pybullet_data
from pybullet_ur5.robot import UR5Robotiq85
from pybullet_utils.bullet_client import BulletClient
import time
import numpy as np

# humanoid
# from deep_mimic.env.motion_capture_data import MotionCaptureData
from deep_mimic.env.motion_capture_data import MotionCaptureData
from deep_mimic.mocap.humanoid_with_rev_xyz import Humanoid
from deep_mimic.mocap.humanoid_with_rev_xyz import HumanoidPose

# Torch imports
import torch

class TrajectoryFollower_H_Clamp():
    def __init__(self, eef_to_cp, right_elbow_joint_to_cp,
                 robot_base_pose, human_arm_lower_limits, human_arm_upper_limits, human_rest_poses, robot_rest_poses):
        # 2nd BC server
        self.bc_second = BulletClient(connection_mode=p.DIRECT)
        self.bc_second.setAdditionalSearchPath(pybullet_data.getDataPath())

        # load humanoid in the 2nd server
        motionPath = 'deep_mimic/mocap/data/Sitting1.json'
        motion = MotionCaptureData()
        motion.Load(motionPath)

        human_base_pos = (0, 0, 0.3)
        human_base_orn = self.bc_second.getQuaternionFromEuler((0, 1.57, 0))
        self.humanoid = Humanoid(self.bc_second, motion, baseShift=human_base_pos, ornShift=human_base_orn)
        self.right_shoulder_y = 3
        self.right_shoulder_p = 4
        self.right_shoulder_r = 5
        self.right_elbow = 7

        # load robot in the 2nd server
        self.robot = UR5Robotiq85(self.bc_second, robot_base_pose[0], robot_base_pose[1])
        self.robot.load()
        self.robot.reset()

        # initialize T_eef_cp (constant)
        self.eef_to_cp = eef_to_cp
        self.cp_to_eef = self.bc_second.invertTransform(self.eef_to_cp[0], self.eef_to_cp[1])

        # initialize T_right_elbow_to_cp (constant)
        self.right_elbow_joint_to_cp = right_elbow_joint_to_cp
        self.cp_to_right_elbow_joint = self.bc_second.invertTransform(self.right_elbow_joint_to_cp[0], self.right_elbow_joint_to_cp[1])

        # human arm joint parameters (for IK)
        self.human_arm_lower_limits = human_arm_lower_limits
        self.human_arm_upper_limits = human_arm_upper_limits
        self.human_arm_joint_ranges = list(np.array(human_arm_upper_limits) - np.array(human_arm_lower_limits))
        self.human_rest_poses = human_rest_poses

        # robot arm joint parameters (for IK)
        self.robot_rest_poses = robot_rest_poses

    def clamp_human_joints(self, q_R):
        # get eef pose from q_R
        for i, joint in enumerate(self.robot.arm_controllable_joints):
            self.bc_second.resetJointState(self.robot.id, joint, q_R[i])
        world_to_eef = self.bc_second.getLinkState(self.robot.id, self.robot.eef_id)[:2]

        # get cp pose
        world_to_cp = self.bc_second.multiplyTransforms(world_to_eef[0], world_to_eef[1],
                                                        self.eef_to_cp[0], self.eef_to_cp[1])
        world_to_right_elbow_joint = self.bc_second.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                                        self.cp_to_right_elbow_joint[0], self.cp_to_right_elbow_joint[1])

        # IK -> get human joint angles
        q_H = self.bc_second.calculateInverseKinematics(self.humanoid._humanoid, self.right_elbow, 
                                                        targetPosition=world_to_right_elbow_joint[0], targetOrientation=world_to_right_elbow_joint[1],
                                                        lowerLimits=self.human_arm_lower_limits, upperLimits=self.human_arm_upper_limits,
                                                        jointRanges=self.human_arm_joint_ranges, restPoses=self.human_rest_poses,
                                                        maxNumIterations=50
                                                        )
        q_H = np.clip(q_H, self.human_arm_lower_limits, self.human_arm_upper_limits)

        # move humanoid in the 2nd server, get new cp pose
        self.bc_second.resetJointState(self.humanoid._humanoid, self.right_shoulder_y, q_H[0])
        self.bc_second.resetJointState(self.humanoid._humanoid, self.right_shoulder_p, q_H[1])
        self.bc_second.resetJointState(self.humanoid._humanoid, self.right_shoulder_r, q_H[2])
        self.bc_second.resetJointState(self.humanoid._humanoid, self.right_elbow, q_H[3])
        world_to_right_elbow_joint = self.bc_second.getLinkState(self.humanoid._humanoid, self.right_elbow)[4:6]
        world_to_cp = self.bc_second.multiplyTransforms(world_to_right_elbow_joint[0], world_to_right_elbow_joint[1],
                                                        self.right_elbow_joint_to_cp[0], self.right_elbow_joint_to_cp[1])

        # get new eef pose
        world_to_eef = self.bc_second.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                                        self.cp_to_eef[0], self.cp_to_eef[1])

        # IK -> get new robot joint angles
        q_R_handshake = self.bc_second.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                        lowerLimits=self.robot.arm_lower_limits, upperLimits=self.robot.arm_upper_limits, 
                                                        jointRanges=self.robot.arm_joint_ranges, restPoses=self.robot_rest_poses,
                                                        maxNumIterations=50)
        q_R_handshake = [q_R_handshake[i] for i in range(len(self.robot.arm_controllable_joints))]

        for i, joint in enumerate(self.robot.arm_controllable_joints):
            self.bc_second.resetJointState(self.robot.id, joint, q_R[i])
        eef_pose = self.bc_second.getLinkState(self.robot.id, self.robot.eef_id)[:2]

        if np.linalg.norm(np.array(world_to_eef[0])-np.array(eef_pose[0])) > 0.05:
            print('traj follower - handshake failed')
            q_R_handshake = q_R
        
        return q_R_handshake