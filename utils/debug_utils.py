from scipy.spatial.transform import Rotation as R
import pybullet as p

def ur5_debug_parameter(env):
        for _ in range(50):
            env.move_human_arm([3.1, 1.57, -1.57, 0])
            env.bc.stepSimulation()

        pos = env.bc.getLinkState(env.humanoid._humanoid, env.right_elbow)[0]
        pos_up = (pos[0], pos[1]+0.2, pos[2])
        print(pos)

        position_control_group = []
        position_control_group.append(env.bc.addUserDebugParameter('x', -1.5, 1.5, pos_up[0]))
        position_control_group.append(env.bc.addUserDebugParameter('y', -1.5, 1.5, pos_up[1]))
        position_control_group.append(env.bc.addUserDebugParameter('z', -1.5, 1.5, pos_up[2]))
        position_control_group.append(env.bc.addUserDebugParameter('roll', -3.14, 3.14, 3.14))
        position_control_group.append(env.bc.addUserDebugParameter('pitch', -3.14, 3.14, 0))
        position_control_group.append(env.bc.addUserDebugParameter('yaw', -3.14, 3.14, -1.57))
        position_control_group.append(env.bc.addUserDebugParameter('gripper_opening', 0, 0.085, 0.08))

        while True:
            env.bc.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

            parameter = []
            for i in range(len(position_control_group)):
                parameter.append(env.bc.readUserDebugParameter(position_control_group[i]))

            env.robot.move_ee(action=parameter[:-1], control_method='end')
            env.robot.move_gripper(parameter[-1])

            # env.move_human_arm([2.7, 1.4, -1.9, 0])

            env.bc.stepSimulation()

            # for i, joint_id in enumerate(env.robot.arm_controllable_joints):
            #     print(i, env.bc.getJointState(env.robot.id, joint_id)[0])

            # right_arm = []
            # right_arm.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_y)[0])
            # right_arm.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_p)[0])
            # right_arm.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_r)[0])
            # right_arm.append(env.bc.getJointState(env.humanoid._humanoid, env.right_elbow)[0])
            # print(right_arm)


def draw_frame(env, position, quaternion=[0, 0, 0, 1]):
        m = R.from_quat(quaternion).as_matrix()
        x_vec = m[:, 0]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for color, column in zip(colors, range(3)):
            vec = m[:, column]
            from_p = position
            to_p = position + (vec * 0.1)
            env.bc.addUserDebugLine(from_p, to_p, color, lineWidth=3, lifeTime=0)

def draw_control_points(env):
    q = [-2.0727, -1.3535,  1.5732, -0.0927, -1.0748, -0.1267]
    control_points = [[ 0.5000,  0.5000,  0.3892],
        [ 0.4736,  0.4518,  0.3892],
        [ 0.5264,  0.5482,  0.3892],
        [ 0.5482,  0.4736,  0.3892],
        [ 0.4347,  0.3809,  0.3892],
        [ 0.4083,  0.3326,  0.3892],
        [ 0.4612,  0.4291,  0.3892],
        [ 0.4119,  0.5299,  0.8042],
        [ 0.4739,  0.5415,  0.8151],
        [ 0.4354,  0.4713,  0.8151],
        [ 0.3883,  0.5884,  0.7933],
        [ 0.3499,  0.5182,  0.7933],
        [ 0.0761,  0.7138,  0.7187],
        [ 0.0728,  0.7156,  0.7484],
        [ 0.0822,  0.7104,  0.6641],
        [ 0.0500,  0.7281,  0.7149],
        [ 0.0314,  0.6322,  0.7187],
        [ 0.0063,  0.6159,  0.7168],
        [ 0.0565,  0.6485,  0.7205],
        [ 0.0253,  0.6356,  0.7732],
        [ 0.0420,  0.6264,  0.6248],
        [ 0.0673,  0.6424,  0.6228],
        [ 0.0166,  0.6105,  0.6267],
        [ 0.0421,  0.6226,  0.5950],
        [ 0.0405,  0.6842,  0.6172],
        [-0.0102,  0.6523,  0.6212],
        [ 0.0153,  0.6644,  0.5894],
        [ 0.0861,  0.5576,  0.6339],
        [ 0.1505,  0.4572,  0.6473],
        [ 0.0608,  0.5416,  0.6359],
        [ 0.1114,  0.5736,  0.6320],
        [ 0.0860,  0.5615,  0.6637],
        [ 0.0863,  0.5537,  0.6042],
        [ 0.0692,  0.4823,  0.6437],
        [ 0.1620,  0.5409,  0.6365],
        [ 0.1041,  0.4279,  0.6509],
        [ 0.1969,  0.4865,  0.6437]]
    
    env.reset_robot(env.robot, q)
    for control_point in control_points:
        env.draw_sphere_marker(position=control_point, radius=0.02)

    print('done')

def draw_control_points_from_pcd(env, pcd):
    for control_point in pcd:
        env.draw_sphere_marker(position=control_point, radius=0.02)

    print('done')

def draw_sphere_marker(env, position, radius=0.07, color=[1, 0, 0, 1]):
    vs_id = env.bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    col_id = env.bc.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    marker_id = env.bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vs_id)
    return marker_id 