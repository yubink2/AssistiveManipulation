import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import pybullet as p
import pybullet_data

from tqdm import tqdm
from pybullet_ur5.robot import Panda, UR5Robotiq85, UR5Robotiq140
from pybullet_ur5.utilities import YCBModels, Camera
import time
import math
import numpy as np

from pybullet_utils.bullet_client import BulletClient
from deep_mimic.env.motion_capture_data import MotionCaptureData

# from humanoid import Humanoid
# from humanoid import HumanoidPose

# from humanoid_with_rev import Humanoid
# from humanoid_with_rev import HumanoidPose

from humanoid_with_rev_xyz import Humanoid
from humanoid_with_rev_xyz import HumanoidPose

from deepmimic_json_generator import *
from transformation import *

import sys
sys.path.append("/usr/lib/python3/dist-packages")



def Reset(humanoid):
  global simTime
  humanoid.Reset()
  simTime = 0
  humanoid.SetSimTime(simTime)
  pose = humanoid.InitializePoseFromMotionData()
  humanoid.ApplyPose(pose, True, True, humanoid._humanoid, bc)

def euler_angles_from_vector(position, center):
    if center[0] > position[0]:
        x,y,z = center-position
    else:
        x,y,z = position-center

    length = math.sqrt(x**2 + y**2 + z**2)
    pitch = math.acos(z/length)
    yaw = math.atan(y/x)
    roll = math.pi if position[0] > center[0] else 0

    euler_angles = [roll,pitch,yaw]
    return euler_angles

def human_motion_from_frame_data(humanoid, utNum, bc_arg):
  keyFrameDuration = motion.KeyFrameDuraction()
  bc_arg.stepSimulation()
  humanoid.RenderReference(utNum * keyFrameDuration, bc_arg)  # RenderReference calls Slerp() & ApplyPose()

def human_motion_from_frame_data_without_applypose(humanoid, utNum, bc_arg):
  keyFrameDuration = motion.KeyFrameDuraction()
  bc_arg.stepSimulation()
  pose = humanoid.RenderReferenceWithoutApplyPose(utNum * keyFrameDuration)
  print('--human_zmotion_from_frame_data_without_applypose: ', pose._rightShoulderRot)
  print('--human_zmotion_from_frame_data_without_applypose: ', pose._rightElbowRot)

def draw_sphere_marker(bc, position, radius=0.07, color=[1, 0, 0, 1]):
  vs_id = bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
  marker_id = bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
  return marker_id

def remove_marker(bc, marker_id):
    bc.removeBody(marker_id)

def reset_human_arm(bc, humanoid, q_human):
  bc.resetJointState(humanoid._humanoid, 3, q_human[0])
  bc.resetJointState(humanoid._humanoid, 4, q_human[1])
  bc.resetJointState(humanoid._humanoid, 5, q_human[2])
  bc.resetJointState(humanoid._humanoid, 7, q_human[3])


# dataset_path = 'data/data_3d_h36m.npz'
# motionPath = 'data/Greeting.json'
# json_path = 'data/Greeting.json'
# action = 'Greeting'
# subject = 'S11'
# fps = 24
# loop = 'wrap'


right_shoulder_y = 3
right_shoulder_p = 4
right_shoulder_r = 5
right_elbow = 7

motionPaths = ['data/Greeting.json', 'data/Eating.json', 'data/Walking.json', 'data/Sitting1.json']
rightElbows = []
rightShouldersY = []
rightShouldersP = []
rightShouldersR = []

for motionPath in motionPaths:
  bc = BulletClient(connection_mode=p.GUI)
  bc.setAdditionalSearchPath(pybullet_data.getDataPath())
  bc.setGravity(0, 0, -9.8) 

  planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0))

  motion = MotionCaptureData()
  motion.Load(motionPath)

  basePos = (0, 0, 0.3)
  # baseOrn = bc.getQuaternionFromEuler((0, -1.57, 0))  # this is the "correct" orientation
  baseOrn = bc.getQuaternionFromEuler((0, 1.57, 0))
  humanoid = Humanoid(bc, motion, basePos, baseOrn)
  print('loaded')

  ## simulating human motion based on frame datasets
  simTime = 0
  keyFrameDuration = motion.KeyFrameDuraction()
  for utNum in range(motion.NumFrames()):
    bc.stepSimulation()
    humanoid.RenderReference(utNum * keyFrameDuration, bc)

  ## collect range of feasible human joints
  for angle in humanoid._rightElbowJointAnglesList:
    rightElbows.append(angle)

  ## _rightShoulderJointAnglesList order: [roll, pitch, yaw]
  ## human arm order: [yaw, pitch, roll]
  for angle1, angle2, angle3 in humanoid._rightShoulderJointAnglesList:
    rightShouldersY.append(angle3)
    rightShouldersP.append(angle2)
    rightShouldersR.append(angle1)

  Reset(humanoid)
  bc.disconnect()

print('rightShouldersYaw min: ', min(rightShouldersY), 'max:', max(rightShouldersY))
print('rightShouldersPitch min: ', min(rightShouldersP), 'max:', max(rightShouldersP))
print('rightShouldersRoll min: ', min(rightShouldersR), 'max:', max(rightShouldersR))
print('rightElbows min: ', min(rightElbows), 'max:', max(rightElbows))

