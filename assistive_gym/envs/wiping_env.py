import os
from gym import spaces
import numpy as np
import pybullet as p
import random

from .env import AssistiveEnv


class WipingEnv(AssistiveEnv):
    def __init__(self):
        super(WipingEnv, self).__init__()
        self.right_shoulder = 6
        self.right_elbow = 7
        self.right_wrist = 8
        self.human_controllable_joints = [3, 4, 5, 7]
        self.human_right_arm = [3, 4, 5, 6, 7, 8]

        # wiping task parameters (from assistive gym config)
        self.robot_forces = 1.0
        self.robot_gains = 0.05
        self.distance_weight = 1.0
        self.action_weight = 0.01
        self.wiping_reward_weight = 5.0
        self.task_success_threshold = 0.6

    def step(self, action):
        self.take_step(action, gains=self.robot_gains, forces=self.robot_forces)

        new_contact_points = self.get_new_contact_points()
        obs = self._get_obs()

        # Penalize getting far away from the targeted human arm
        if 'upperarm' in self.arm_side:
            closest_points = self.bc.getClosestPoints(bodyA=self.tool, bodyB=self.humanoid._humanoid, distance=100.0, 
                                                    linkIndexA=1, linkIndexB=self.right_shoulder, physicsClientId=self.bc._client)
        elif 'forearm' in self.arm_side:
            closest_points = self.bc.getClosestPoints(bodyA=self.tool, bodyB=self.humanoid._humanoid, distance=100.0, 
                                                    linkIndexA=1, linkIndexB=self.right_elbow, physicsClientId=self.bc._client)
        if closest_points:
            reward_distance = -min([c[8] for c in closest_points])
        else:
            reward_distance = -1.0

        # Penalize actions
        reward_action = -np.sum(np.square(action))  

        # Reward new contact points on a person
        reward_new_contact_points = new_contact_points

        reward = self.distance_weight*reward_distance + self.action_weight*reward_action + self.wiping_reward_weight*reward_new_contact_points

        info = {'number_of_targets_cleared': self.task_success, 'total_target_count': self.total_target_count, 'feasible_target_count': self.feasible_targets_count, 
                'task_success': int(self.task_success >= (self.feasible_targets_count*self.task_success_threshold)), 
                'task_percentage': self.task_success/self.feasible_targets_count*100,
                'action_robot_len': self.action_robot_len, 'obs_robot_len': self.obs_robot_len}
        done = bool(self.task_success >= (self.feasible_targets_count*self.task_success_threshold))

        # print(f'reward_distance: {reward_distance}, reward: {reward}')

        return obs, reward, done, info
    
    def get_new_contact_points(self):
        new_contact_points = 0
        for c in self.bc.getContactPoints(bodyA=self.tool, bodyB=self.humanoid._humanoid, physicsClientId=self.bc._client):
            linkA = c[3]
            linkB = c[4]
            contact_position = np.array(c[6])  # contact position on B
            if linkA in [1]:  # tool endtip
                # Only consider contact with human upperarm, forearm, hand
                if linkB < 0 or linkB not in self.human_right_arm:
                    continue

                # Check feasible targets
                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.feasible_targets_pos_world, self.feasible_targets)):
                    if np.linalg.norm(contact_position - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm 
                        new_contact_points += 1
                        self.task_success += 1
                        self.bc.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.bc._client)
                        indices_to_delete.append(i)
                self.feasible_targets_pos = [t for i, t in enumerate(self.feasible_targets_pos) if i not in indices_to_delete]
                self.feasible_targets = [t for i, t in enumerate(self.feasible_targets) if i not in indices_to_delete]
                self.feasible_targets_pos_world = [t for i, t in enumerate(self.feasible_targets_pos_world) if i not in indices_to_delete]

        return new_contact_points

    def _get_obs(self):
        obs = np.ones(self.obs_robot_len)*30  # padding irrelevant obs values to 30
        real_obs_len = 3 + 4 + 6 + len(self.feasible_targets_pos_world)*3

        tool_pos, tool_orn = self.bc.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        tool_pos = np.array(tool_pos)
        tool_orn = np.array(tool_orn)

        robot_joint_states = self.bc.getJointStates(self.robot_2.id, jointIndices=self.robot_2.arm_controllable_joints, physicsClientId=self.bc._client)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])

        real_obs = np.concatenate([tool_pos, tool_orn, robot_joint_positions] + list(self.feasible_targets_pos_world)).ravel().astype(np.float32)
        obs[:real_obs_len] = real_obs

        return obs

    def reset(self):
        self.bc.resetSimulation(self.bc._client)
        self.setup_timing()
        self.create_world()

        self.task_success = 0
        self.contact_points_on_arm = {}
        self.robot_lower_limits = self.robot_2.arm_lower_limits
        self.robot_upper_limits = self.robot_2.arm_upper_limits

        feasible_targets_found = False
        while True:
            # randomize human arm config and check for collisions
            self.robot_2.reset()
            self.reset_and_check()
            
            # ###
            # # q_H = [3.0, 0, -1.9, 1.0]
            # q_H = [1.7022014187018732, 0.8518481218916178, -2.0389745750478308, 1.1132988114423075]
            # self.reset_human_arm(q_H)
            # self.lock_human_joints(q_H)
        
            feasible_targets_found = self.generate_targets()

            if not feasible_targets_found:
                self.remove_targets()
                continue

            # mark feasible targets
            if self.gui:
                self.mark_feasible_targets()

            # initialize tool & compute eef_to_tool
            self.robot_2.reset()
            self.init_tool()

            # reset robot to init config
            for i, joint_id in enumerate(self.robot_2.arm_controllable_joints):
                self.bc.resetJointState(self.robot_2.id, joint_id, self.init_q_R[i], physicsClientId=self.bc._client)

            # reset tool and attach it to eef
            world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
            world_to_tool = self.bc.multiplyTransforms(world_to_eef[0], world_to_eef[1],
                                                    self.eef_to_tool[0], self.eef_to_tool[1], physicsClientId=self.bc._client)
            self.bc.resetBasePositionAndOrientation(self.tool, world_to_tool[0], world_to_tool[1], physicsClientId=self.bc._client)
            self.attach_tool()

            print(f'init_q_R: {self.init_q_R}')

            if feasible_targets_found:
                break

        return self._get_obs()

    def generate_targets(self):
        self.target_indices_to_ignore = []
        self.upperarm_length, self.upperarm_radius = 0.144000+0.036000, 0.036000
        self.forearm_length, self.forearm_radius = 0.108000+0.028000*2, 0.028000

        # generate capsule points
        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), 
                                                                radius=self.upperarm_radius, distance_between_points=0.03)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, -0.01]), p2=np.array([0, 0, -self.forearm_length-0.01]), 
                                                                radius=self.forearm_radius, distance_between_points=0.03)
        
        # create target points
        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.bc._client)
        self.targets_upperarm = []
        self.targets_forearm = []
        for _ in range(len(self.targets_pos_on_upperarm)):
            self.targets_upperarm.append(self.bc.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, 
                                                           basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.bc._client))
        for _ in range(len(self.targets_pos_on_forearm)):
            self.targets_forearm.append(self.bc.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, 
                                                          basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.bc._client))
        self.total_target_count = len(self.targets_upperarm) + len(self.targets_forearm)

        # move targets to initial positions
        self.update_targets()

        # feasible targets
        feasible_targets_found = self.get_feasible_targets_pos()

        return feasible_targets_found
    
    def remove_targets(self):
        for target in self.targets_upperarm:
            self.bc.removeBody(target, physicsClientId=self.bc._client)
        for target in self.targets_forearm:
            self.bc.removeBody(target, physicsClientId=self.bc._client)

    def update_targets(self):
        upperarm_pos, upperarm_orient = self.bc.getLinkState(self.humanoid._humanoid, self.right_shoulder, computeForwardKinematics=True, physicsClientId=self.bc._client)[4:6]
        upperarm_orient = self.util.rotate_quaternion_by_axis(upperarm_orient, axis='x', degrees=-90)
        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            target_pos, target_orn = self.bc.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.bc._client)
            self.targets_pos_upperarm_world.append(target_pos)
            self.bc.resetBasePositionAndOrientation(target, target_pos, target_orn, physicsClientId=self.bc._client)

        forearm_pos, forearm_orient = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow, computeForwardKinematics=True, physicsClientId=self.bc._client)[4:6]
        forearm_orient = self.util.rotate_quaternion_by_axis(forearm_orient, axis='x', degrees=-90)
        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos, target_orn = self.bc.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.bc._client)
            self.targets_pos_forearm_world.append(target_pos)
            self.bc.resetBasePositionAndOrientation(target, target_pos, target_orn, physicsClientId=self.bc._client)

    def mark_feasible_targets(self):
        # change color for feasible targets
        for target in self.feasible_targets:
            self.bc.changeVisualShape(target, -1, rgbaColor=[0, 0.2, 1, 1], physicsClientId=self.bc._client)

    def get_feasible_targets_pos(self):
        # randomize order of trial
        self.target_order_flags = {'upperarm_front': False,
                                   'upperarm_back': False,
                                   'forearm_front': False,
                                   'forearm_back': False}
        self.target_trial_order = ['upperarm_front', 'upperarm_back', 'forearm_front', 'forearm_back']
        self.target_trial_order = random.sample(self.target_trial_order, len(self.target_trial_order))
        self.target_axis_trial_order = [0, 1]
        self.target_axis_trial_order = random.sample(self.target_axis_trial_order, len(self.target_axis_trial_order))

        # flag
        feasible_targets_found = False

        # split targets to front and back
        def split_targets(targets_pos, axis):
            """Splits targets into front and back halves based on the x or y coordinate. (axis = 0 or 1)"""
            z_positions = np.array([pos[axis] for pos in targets_pos])
            mid_point = np.median(z_positions)
            front_indices = [i for i, pos in enumerate(targets_pos) if pos[axis] > mid_point]
            back_indices = [i for i, pos in enumerate(targets_pos) if pos[axis] <= mid_point]
            return front_indices, back_indices

        def split_half(targets_pos, axis=0):
            """Split targets into front and back, returning indices."""
            front_half_indices, back_half_indices = split_targets(targets_pos, axis)
            return front_half_indices, back_half_indices

        # check both x and y axis
        for axis in self.target_axis_trial_order:
            front_targets_on_upperarm_indices, back_targets_on_upperarm_indices = split_half(self.targets_pos_on_upperarm, axis)
            front_targets_on_forearm_indices, back_targets_on_forearm_indices = split_half(self.targets_pos_on_forearm, axis)
            self.target_pos_dict = {'upperarm_front': np.array(self.targets_pos_on_upperarm)[front_targets_on_upperarm_indices],
                                    'upperarm_back': np.array(self.targets_pos_on_upperarm)[back_targets_on_upperarm_indices],
                                    'forearm_front': np.array(self.targets_pos_on_forearm)[front_targets_on_forearm_indices],
                                    'forearm_back': np.array(self.targets_pos_on_forearm)[back_targets_on_forearm_indices]}
            self.target_pos_world_dict = {'upperarm_front': np.array(self.targets_pos_upperarm_world)[front_targets_on_upperarm_indices],
                                          'upperarm_back': np.array(self.targets_pos_upperarm_world)[back_targets_on_upperarm_indices],
                                          'forearm_front': np.array(self.targets_pos_forearm_world)[front_targets_on_forearm_indices],
                                          'forearm_back': np.array(self.targets_pos_forearm_world)[back_targets_on_forearm_indices]}
            self.target_dict = {'upperarm_front': np.array(self.targets_upperarm)[front_targets_on_upperarm_indices],
                                'upperarm_back': np.array(self.targets_upperarm)[back_targets_on_upperarm_indices],
                                'forearm_front': np.array(self.targets_forearm)[front_targets_on_forearm_indices],
                                'forearm_back': np.array(self.targets_forearm)[back_targets_on_forearm_indices]}
            
            for order_key in self.target_trial_order:
                if order_key=='upperarm_back' and axis==0:
                    continue    

                # reset robot
                self.robot_2.reset()

                # set flag & targets
                self.target_order_flags[order_key] = True
                targets_pos = self.target_pos_dict[order_key]
                targets_pos_world = self.target_pos_world_dict[order_key]
                targets = self.target_dict[order_key]

                # compute world_to_target_point
                def compute_mean_target(targets_pos):
                    targets_pos = np.array(targets_pos)
                    mean_targets_pos = np.mean(targets_pos, axis=0)
                    return tuple(mean_targets_pos)

                if 'upperarm' in order_key:
                    upperarm_orient = self.bc.getLinkState(self.humanoid._humanoid, self.right_shoulder, computeForwardKinematics=True, physicsClientId=self.bc._client)[5]
                    self.world_to_target_point = [compute_mean_target(targets_pos_world), upperarm_orient]
                else:
                    forearm_orient = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow, computeForwardKinematics=True, physicsClientId=self.bc._client)[5]
                    self.world_to_target_point = [compute_mean_target(targets_pos_world), forearm_orient]
                
                if 'front' in order_key and axis==1:
                    self.world_to_target_point = [self.world_to_target_point[0],
                                                  self.util.rotate_quaternion_by_axis(self.world_to_target_point[1], axis='y', degrees=90)]
                elif 'back' in order_key and axis==0:
                    self.world_to_target_point = [self.world_to_target_point[0],
                                                  self.util.rotate_quaternion_by_axis(self.world_to_target_point[1], axis='y', degrees=180)]
                elif 'back' in order_key and axis==1:
                    self.world_to_target_point = [self.world_to_target_point[0],
                                                  self.util.rotate_quaternion_by_axis(self.world_to_target_point[1], axis='y', degrees=270)]

                # compute desired world_to_eef (initial robot config)
                world_to_eef = self.bc.multiplyTransforms(self.world_to_target_point[0], self.world_to_target_point[1],
                                                          self.target_to_eef[0], self.target_to_eef[1], physicsClientId=self.bc._client)
                
                # if self.gui:
                #     self.util.draw_frame(world_to_eef[0], world_to_eef[1])  #####

                # set robot initial joint state
                q_robot_2 = self.bc.calculateInverseKinematics(self.robot_2.id, self.robot_2.eef_id, world_to_eef[0], world_to_eef[1],
                                                               lowerLimits=self.robot_2.arm_lower_limits, upperLimits=self.robot_2.arm_upper_limits, 
                                                               jointRanges=self.robot_2.arm_joint_ranges, restPoses=self.robot_2.arm_rest_poses,
                                                               maxNumIterations=40, physicsClientId=self.bc._client)
                q_robot_2 = [q_robot_2[i] for i in range(len(self.robot_2.arm_controllable_joints))]
                if min(q_robot_2) < min(self.robot_2.arm_lower_limits) or max(q_robot_2) > max(self.robot_2.arm_upper_limits):  # invalid joint state
                    # reset flag
                    self.target_order_flags[order_key] = False
                    continue

                for i, joint_id in enumerate(self.robot_2.arm_controllable_joints):
                    self.bc.resetJointState(self.robot_2.id, joint_id, q_robot_2[i], physicsClientId=self.bc._client)
                self.bc.stepSimulation(physicsClientId=self.bc._client)

                # check if config is valid
                eef_pos = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[0]
                dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pos))
                if dist > 0.03 or self.robot_2_in_collision(q_robot_2):
                    # reset flag
                    self.target_order_flags[order_key] = False
                    continue

                #####
                # compute desired world_to_eef (check if it can get closer to the target point)
                world_to_eef = self.bc.multiplyTransforms(self.world_to_target_point[0], self.world_to_target_point[1],
                                                          self.target_closer_to_eef[0], self.target_closer_to_eef[1], physicsClientId=self.bc._client)

                # if self.gui:
                #     self.util.draw_frame(world_to_eef[0], world_to_eef[1])  #####

                # set robot initial joint state
                q_robot_2_closer = self.bc.calculateInverseKinematics(self.robot_2.id, self.robot_2.eef_id, world_to_eef[0], world_to_eef[1],
                                                               lowerLimits=self.robot_2.arm_lower_limits, upperLimits=self.robot_2.arm_upper_limits, 
                                                               jointRanges=self.robot_2.arm_joint_ranges, restPoses=self.robot_2.arm_rest_poses,
                                                               maxNumIterations=40, physicsClientId=self.bc._client)
                q_robot_2_closer = [q_robot_2_closer[i] for i in range(len(self.robot_2.arm_controllable_joints))]
                if min(q_robot_2_closer) < min(self.robot_2.arm_lower_limits) or max(q_robot_2_closer) > max(self.robot_2.arm_upper_limits):  # invalid joint state
                    # reset flag
                    self.target_order_flags[order_key] = False
                    continue

                for i, joint_id in enumerate(self.robot_2.arm_controllable_joints):
                    self.bc.resetJointState(self.robot_2.id, joint_id, q_robot_2_closer[i], physicsClientId=self.bc._client)
                self.bc.stepSimulation(physicsClientId=self.bc._client)

                # check if config is valid
                eef_pos = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[0]
                dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pos))
                if dist > 0.03 or self.robot_2_in_collision(q_robot_2_closer):
                    # reset flag
                    self.target_order_flags[order_key] = False
                    continue
                #####

                self.feasible_targets_pos = targets_pos
                self.feasible_targets_pos_world = targets_pos_world
                self.feasible_targets = targets
                self.feasible_targets_count = len(targets)
                self.init_q_R = q_robot_2
                self.arm_side = order_key
                feasible_targets_found = True
                print(f'arm_side: {self.arm_side}, axis: {axis}')
                break

            if feasible_targets_found:
                break

        return feasible_targets_found
