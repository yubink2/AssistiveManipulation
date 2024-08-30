import os, inspect
import numpy as np
import time

import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

# humanoid
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
parentparentparentdir = os.path.dirname(parentparentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(1, parentparentdir)
os.sys.path.insert(1, parentparentparentdir)
from deep_mimic.env.motion_capture_data import MotionCaptureData
from humanoid_with_rev_xyz import Humanoid

# robot
from pybullet_ur5.robot import UR5Robotiq85
from utils.collision_utils import get_collision_fn
from .util import Util
from .targets_util import TargetsUtil

class WipingDemo():
    def __init__(self):
        # Start the bullet physics server
        self.bc = BulletClient(connection_mode=p.GUI)
        self.util = Util(self.bc, self.np_random)
        self.targets_util = TargetsUtil(self.bc, self.util)

    def reset(self):
        self.create_world()
        self.targets_util.generate_new_targets_pos()
        self.targets_util.generate_targets()

    def create_world(self):
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc.setGravity(0, 0, 0, physicsClientId=self.bc._client) 

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, 0, -0.04), physicsClientId=self.bc._client)
        bed_id = self.bc.loadURDF("./urdf/bed_0.urdf", (0.0, -0.1, 0.0), useFixedBase=True, physicsClientId=self.bc._client)

        # load human
        human_base_pos = (0, 0, 0.3)
        human_base_orn = self.bc.getQuaternionFromEuler((0, 1.57, 0))
        motionPath = 'deep_mimic/mocap/data/Sitting1.json'
        self.motion = MotionCaptureData()
        self.motion.Load(motionPath)
        self.humanoid = Humanoid(self.bc, self.motion, baseShift=human_base_pos, ornShift=human_base_orn)

        # load second robot
        self.robot_2_base_pose = ((0.65, 0, 0.25), (0, 0, -1.57))
        cube_id = p.loadURDF("./urdf/cube_0.urdf", 
                            (self.robot_2_base_pose[0][0], self.robot_2_base_pose[0][1], self.robot_2_base_pose[0][2]-0.15), useFixedBase=True,
                            physicsClientId=self.bc._client)
        self.robot_2 = UR5Robotiq85(self.bc, self.robot_2_base_pose[0], self.robot_2_base_pose[1])
        self.robot_2.load()
        self.robot_2.reset()

        self.bed_id = bed_id
        self.cube_id = cube_id
        self.targets_pos_on_upperarm = None
        self.targets_pos_on_forearm = None

        # initialize collision checker
        obstacles = [self.bed_id, self.cube_id, self.humanoid._humanoid]
        self.robot_2_in_collision = get_collision_fn(self.robot_2.id, self.robot_2.arm_controllable_joints, obstacles=obstacles,
                                                     attachments=[], self_collisions=True,
                                                     disabled_collisions=set())
        
        # compute target_to_eef & target_closer_to_eef
        world_to_eef = self.bc.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        target_orn = self.util.rotate_quaternion_by_axis(world_to_eef[1], axis='z', degrees=180)
        world_to_target = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.3], target_orn]
        world_to_target_closer = [[world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.2], target_orn]

        self.target_orn = target_orn
        target_to_world = self.bc.invertTransform(world_to_target[0], world_to_target[1], physicsClientId=self.bc._client)
        target_closer_to_world = self.bc.invertTransform(world_to_target_closer[0], world_to_target_closer[1], physicsClientId=self.bc._client)
        self.target_to_eef = self.bc.multiplyTransforms(target_to_world[0], target_to_world[1],
                                                        world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        self.target_closer_to_eef = self.bc.multiplyTransforms(target_closer_to_world[0], target_closer_to_world[1],
                                                               world_to_eef[0], world_to_eef[1], physicsClientId=self.bc._client)
        

if __name__ == '__main__':
    env = WipingDemo()
    env.reset()
    print('')