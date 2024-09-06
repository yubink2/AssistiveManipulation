import os, inspect
import numpy as np
import time

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

class WipingDemo():
    def __init__(self, gui=False):
        # Start the bullet physics server
        self.gui = gui
        if self.gui:
            self.bc = BulletClient(connection_mode=p.GUI)
        else:
            self.bc = BulletClient(connection_mode=p.DIRECT)
        self.util = Util(self.bc)
        self.targets_util = TargetsUtil(self.bc._client, self.util)

    def reset(self):
        self.create_world()
        self.init_tool()
        self.targets_util.init_targets_util(self.humanoid._humanoid, self.right_shoulder, self.right_elbow, self.human_right_arm,
                                            self.robot_2, self.tool,
                                            self.target_to_eef, self.target_closer_to_eef, self.robot_2_in_collision)
        
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
        self.bed_id = self.bc.loadURDF("./urdf/bed_0.urdf", (0.0, -0.1, 0.0), useFixedBase=True, physicsClientId=self.bc._client)

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

        # load first robot (manipulation)
        self.robot_base_pose = ((0.65, 0.7, 0.25), (0, 0, -1.57))
        self.cube_id = self.bc.loadURDF("./urdf/cube_0.urdf", 
                                   (self.robot_base_pose[0][0], self.robot_base_pose[0][1], self.robot_base_pose[0][2]-0.15), useFixedBase=True)
        self.robot = UR5Robotiq85(self.bc, self.robot_base_pose[0], self.robot_base_pose[1])
        self.robot.load()
        for _ in range(50):
            self.robot.reset()
            self.robot.open_gripper()

        # load second robot (wiping)
        self.robot_2_base_pose = ((0.65, 0, 0.25), (0, 0, -1.57))
        self.cube_2_id = p.loadURDF("./urdf/cube_0.urdf", 
                            (self.robot_2_base_pose[0][0], self.robot_2_base_pose[0][1], self.robot_2_base_pose[0][2]-0.15), useFixedBase=True,
                            physicsClientId=self.bc._client)
        self.robot_2 = UR5Robotiq85(self.bc, self.robot_2_base_pose[0], self.robot_2_base_pose[1])
        self.robot_2.load()
        self.robot_2.reset()

        self.targets_pos_on_upperarm = None
        self.targets_pos_on_forearm = None

        # initialize collision checker
        obstacles = [self.bed_id, self.humanoid._humanoid, self.robot.id, self.cube_id]
        self.robot_2_in_collision = get_collision_fn(self.robot_2.id, self.robot_2.arm_controllable_joints, obstacles=obstacles,
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
            # check for invalid order of target sequence
            if prev_target_pos_world is not None:
                dist = np.linalg.norm(np.array(prev_target_pos_world) - np.array(target_pos_world))
                if dist > 0.035:
                    break

            # compute desired world_to_eef (check if it can get closer to the target point)
            world_to_eef = self.bc.multiplyTransforms(target_pos_world, target_orn_world,
                                                      self.target_closer_to_eef[0], self.target_closer_to_eef[1], physicsClientId=self.bc._client)

            # set robot initial joint state
            q_robot_2_closer = self.bc.calculateInverseKinematics(self.robot_2.id, self.robot_2.eef_id, world_to_eef[0], world_to_eef[1],
                                                            lowerLimits=self.robot_2.arm_lower_limits, upperLimits=self.robot_2.arm_upper_limits, 
                                                            jointRanges=self.robot_2.arm_joint_ranges, restPoses=self.robot_2.arm_rest_poses,
                                                            maxNumIterations=40, physicsClientId=self.bc._client)
            q_robot_2_closer = [q_robot_2_closer[i] for i in range(len(self.robot_2.arm_controllable_joints))]
            q_robot_2_closer = np.clip(q_robot_2_closer, self.robot_2.arm_lower_limits, self.robot_2.arm_upper_limits)

            for i, joint_id in enumerate(self.robot_2.arm_controllable_joints):
                self.bc.resetJointState(self.robot_2.id, joint_id, q_robot_2_closer[i], physicsClientId=self.bc._client)
            self.bc.stepSimulation(physicsClientId=self.bc._client)

            # check if config is valid
            eef_pos = p.getLinkState(self.robot_2.id, self.robot_2.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[0]
            dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pos))
            if not self.robot_2_in_collision(q_robot_2_closer) and dist < 0.035:
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

    def lock_human_joints(self, q_human):
        # Make all joints on the person static by setting mass of each link (joint) to 0
        for j in range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)):
            self.bc.changeDynamics(self.humanoid._humanoid, j, mass=0, physicsClientId=self.bc._client)
        # Set arm joints velocities to 0
        for i, j in enumerate(self.human_controllable_joints):
            self.bc.resetJointState(self.humanoid._humanoid, jointIndex=j, targetValue=q_human[i], targetVelocity=0, physicsClientId=self.bc._client)

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
            if (point[2] in [self.bed_id, self.cube_id, self.cube_2_id, self.robot.id, self.robot_2.id]):
                return True
        return False

    def get_valid_q_H(self):
        """Reset the human arm and check for collisions until no collision is detected."""
        for _ in range(1000):
            q_H = self.generate_random_q_H()
            self.reset_human_arm(q_H)
            self.bc.stepSimulation(physicsClientId=self.bc._client)
            if not self.human_in_collision():
                self.lock_human_joints(q_H)
                world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
                return q_H, world_to_right_elbow
        raise ValueError('valid human config not found!')

    def lock_robot_arm_joints(self, robot, q_robot):
        # Make all joints static by setting mass of each link (joint) to 0
        for j in range(self.bc.getNumJoints(robot.id, physicsClientId=self.bc._client)):
            self.bc.changeDynamics(robot.id, j, mass=0, physicsClientId=self.bc._client)
        # Set arm joints velocities to 0
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(self.robot.id, jointIndex=joint_id, targetValue=q_robot[i], targetVelocity=0, physicsClientId=self.bc._client)

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

if __name__ == '__main__':

    env = WipingDemo(gui=True)
    env.reset()
    env.lock_robot_arm_joints(env.robot, env.robot.arm_rest_poses)

    q_Hs = [[2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409],
            [0, 0, -1.6, 1.0],
            [1.8, 1.0, -2.6, 1.5]]
    arms = ['upperarm', 'forearm']
    total_targets_cleared = 0

    for q_H in q_Hs:
        for arm in arms:
            feasible_targets_found = env.reset_wiping_setup(q_H, arm)
            if not feasible_targets_found:
                print(f'{q_H}, {arm}, feasible targets not found!')
                continue
            
            robot_traj = env.compute_feasible_targets_robot_traj()
            if len(robot_traj) == 0:
                continue
            # robot_traj = env.interpolate_trajectory(robot_traj, alpha=0.5)
            # robot_traj = env.interpolate_trajectory(robot_traj, alpha=0.25)

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
                time.sleep(0.3)
            print(f'targets_cleared: {targets_cleared}, total_targets_cleared: {total_targets_cleared}')
            print(f'wiping {arm} is done')

            env.targets_util.remove_targets()
            env.targets_util.unmark_feasible_targets()
            env.targets_util.update_targets()
            env.detach_tool()

            time.sleep(2)
    
    print('done')