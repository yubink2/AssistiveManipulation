import numpy as np
import pybullet as p

class TargetsUtil:
    def __init__(self, pid, util):
        self.pid = pid
        self.util = util

    def generate_new_targets_pos(self):
        self.target_indices_to_ignore = []
        self.upperarm_length, self.upperarm_radius = 0.144000+0.036000, 0.036000
        self.forearm_length, self.forearm_radius = 0.108000+0.028000*2, 0.028000

        # generate capsule points
        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), 
                                                                radius=self.upperarm_radius, distance_between_points=0.03)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, -0.01]), p2=np.array([0, 0, -self.forearm_length-0.01]), 
                                                                radius=self.forearm_radius, distance_between_points=0.03)
        
    def generate_targets(self):
        assert self.targets_pos_on_upperarm is not None
        assert self.targets_pos_on_forearm is not None

        # create target points
        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.pid)
        self.targets_upperarm = []
        self.targets_forearm = []
        for _ in range(len(self.targets_pos_on_upperarm)):
            self.targets_upperarm.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, 
                                                           basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.pid))
        for _ in range(len(self.targets_pos_on_forearm)):
            self.targets_forearm.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, 
                                                          basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.pid))
        self.total_target_count = len(self.targets_upperarm) + len(self.targets_forearm)

        # move targets to initial positions
        self.update_targets()

        # # feasible targets
        # feasible_targets_found = self.get_feasible_targets_pos()

        # return feasible_targets_found
    
    def remove_targets(self):
        for target in self.targets_upperarm:
            p.removeBody(target, physicsClientId=self.pid)
        for target in self.targets_forearm:
            p.removeBody(target, physicsClientId=self.pid)

    def update_targets(self):
        upperarm_pos, upperarm_orient = p.getLinkState(self.humanoid._humanoid, self.right_shoulder, computeForwardKinematics=True, physicsClientId=self.pid)[4:6]
        upperarm_orient = self.util.rotate_quaternion_by_axis(upperarm_orient, axis='x', degrees=-90)
        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            target_pos, target_orn = p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.pid)
            self.targets_pos_upperarm_world.append(target_pos)
            p.resetBasePositionAndOrientation(target, target_pos, target_orn, physicsClientId=self.pid)

        forearm_pos, forearm_orient = p.getLinkState(self.humanoid._humanoid, self.right_elbow, computeForwardKinematics=True, physicsClientId=self.pid)[4:6]
        forearm_orient = self.util.rotate_quaternion_by_axis(forearm_orient, axis='x', degrees=-90)
        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos, target_orn = p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.pid)
            self.targets_pos_forearm_world.append(target_pos)
            p.resetBasePositionAndOrientation(target, target_pos, target_orn, physicsClientId=self.pid)