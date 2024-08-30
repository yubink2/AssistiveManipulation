import numpy as np
import random
import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AntipodalGraspSampler:
    def __init__(self, friction_coef=0.1, num_cone_faces=8, obj_inward_vector=None,
                 max_num_surface_points=100, num_samples=10, min_contact_dist=0.01):
        self.friction_coef = friction_coef
        self.num_cone_faces = num_cone_faces
        self.max_num_surface_points = max_num_surface_points
        self.num_samples = num_samples
        self.min_contact_dist = min_contact_dist
        self.obj_inward_vector = obj_inward_vector

    def sample_from_cone(self, cone, num_samples=1):
        num_faces = cone.shape[1]
        v_samples = np.empty((num_samples, 3)) 
        for i in range(num_samples):
            lambdas = np.random.gamma(1.0, 1.0, num_faces)
            lambdas = lambdas / np.sum(lambdas)
            v_sample = lambdas * cone
            v_samples[i, :] = np.sum(v_sample, 1)
        return v_samples

    def within_cone(self, cone, n, v):
        v = v / np.linalg.norm(v)
        n = n / np.linalg.norm(n)

        # ensure correct direction
        if (np.dot(v, cone) < 0).any():
            v = -v

        dot_product = np.clip(np.dot(n, v), -1.0, 1.0)
        alpha = np.arccos(dot_product)  # in radians

        # whether alpha is within the friction cone
        return alpha <= np.arctan(self.friction_coef), alpha

    def generate_grasps(self, pc_ply, resolution=0.01, vis=True):
        grasps = []
        pc_ply = self.shuffle_pc_ply(pc_ply)

        point_cloud = np.asarray(pc_ply.points)
        normals = np.asarray(pc_ply.normals)
        indices = np.arange(len(point_cloud))
        
        surface_indices = indices[:min(self.max_num_surface_points, len(point_cloud))]
        surface_points = point_cloud[surface_indices]
        surface_normals = normals[surface_indices]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o')

        for idx in range(len(surface_points)):
            x_surf = surface_points[idx]
            n_surf = surface_normals[idx]

            for _ in range(self.num_samples):
                x1 = x_surf
                n1 = n_surf
                c1 = {'point': x_surf, 'normal': n_surf}
                if c1 is None:
                    continue

                cone1, n1 = self.compute_friction_cone(c1)
                if cone1 is None:
                    continue

                v_samples = self.sample_from_cone(cone1)

                for v in v_samples:
                    c2 = self.find_contact_along_axis(x1, -v, point_cloud, normals)
                    if c2 is None:
                        continue

                    x2 = c2['point']
                    if np.linalg.norm(x1 - x2) < self.min_contact_dist:
                        continue

                    in_cone1, _ = self.within_cone(cone1, n1, v)
                    cone2, n2 = self.compute_friction_cone(c2)
                    if cone2 is None:
                        continue

                    in_cone2, _ = self.within_cone(cone2, n2, -v)


                    if vis:
                        ax.scatter(x1[0], x1[1], x1[2], c='r', marker='o', s=100)
                        ax.scatter(x2[0], x2[1], x2[2], c='g', marker='o', s=100)
                        # ax.quiver(x1[0], x1[1], x1[2], v[0], v[1], v[2], length=0.05, color='r')
                        # ax.quiver(x1[0], x1[1], x1[2], n1[0], n1[1], n1[2], length=0.05, color='b')
                        ax.plot([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]], c='k')
                        # plt.show()

                    if in_cone1 and in_cone2:
                        grasps.append(self.compute_grasp_matrix(c1=c1, c2=c2, grasp_axis=-v))

        if vis:
            plt.show()

        random.shuffle(grasps) 
        return grasps

    def get_contact(self, point, point_cloud):
        normal = self.estimate_normal(point, point_cloud)
        if normal is None:
            return None
        return {'point': point, 'normal': normal}

    def compute_friction_cone(self, contact):
        n = contact['normal']
        if n is None:
            return None, None
        
        angles = np.linspace(0, 2 * np.pi, self.num_cone_faces, endpoint=False)
        cone = np.array([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
        cone = cone * self.friction_coef + n[:, np.newaxis]

        return cone, n
    
    def shuffle_pc_ply(self, pc_ply):
        points = np.asarray(pc_ply.points)
        normals = np.asarray(pc_ply.normals)
        indices = np.arange(points.shape[0])
        np.random.shuffle(indices)

        shuffled_points = points[indices]
        shuffled_normals = normals[indices]

        pc_ply.points = o3d.utility.Vector3dVector(shuffled_points)
        pc_ply.normals = o3d.utility.Vector3dVector(shuffled_normals)

        return pc_ply

    def find_contact_along_axis(self, start_point, axis, point_cloud, normals, max_distance=0.1):
        axis = axis / np.linalg.norm(axis)
        distances = np.linalg.norm(point_cloud - start_point, axis=1)
        within_range_indices = np.where((distances > 0) & (distances < max_distance))[0]

        if len(within_range_indices) == 0:
            return None

        max_distance = -np.inf
        furthest_contact_point = None
        furthest_contact_point_normal = None

        for index in within_range_indices:
            point = point_cloud[index]
            direction = point - start_point
            direction = direction / np.linalg.norm(direction)  # Normalize the direction vector

            if np.dot(direction, axis) > 0.99:  # Check if the direction is almost parallel to the axis
                distance = np.linalg.norm(point - start_point)
                if distance > max_distance:
                    max_distance = distance
                    furthest_contact_point = point
                    furthest_contact_point_normal = normals[index]

        if furthest_contact_point is None:
            return None

        return {'point': furthest_contact_point, 'normal': furthest_contact_point_normal}
    
    def compute_grasp_matrix(self, c1, c2, grasp_axis, gripper_width=0.085, finger_length=0.038):
        grasp_center = (c1['point'] + c2['point']) / 2
        grasp_axis_g = grasp_axis

        if self.obj_inward_vector is None:
            temp_vector = np.array([1.0, 0.0, 0.0])
            if np.allclose(grasp_axis_g, temp_vector):
                temp_vector = np.array([0.0, 1.0, 0.0])

            grasp_axis_b = np.cross(grasp_axis_g, temp_vector)  # grasp approach vector
            grasp_axis_b = grasp_axis_b / np.linalg.norm(grasp_axis_b)
            grasp_axis_r = np.cross(grasp_axis_g, grasp_axis_b)  # grasp binormal vector
        
        else:
            grasp_axis_b = self.obj_inward_vector
            grasp_axis_r = np.cross(grasp_axis_g, grasp_axis_b)

        # 3 vectors --> rotation matrix
        R = np.column_stack((grasp_axis_r, grasp_axis_g, grasp_axis_b))

        grasp_transformation_matrix = np.eye(4)
        grasp_transformation_matrix[:3, :3] = R
        grasp_transformation_matrix[:3, 3] = grasp_center

        return grasp_transformation_matrix
    
###    
def visualize_normals(point_cloud, k=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o')

    sampler = AntipodalGraspSampler()
    for point in point_cloud[:50]:
        normal = sampler.estimate_normal(point, point_cloud, k)
        if normal is not None:
            ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=0.05, color='r')

    plt.show()

def test_antipodal_grasp_sampling(pc_ply, vis=False):
    sampler = AntipodalGraspSampler()
    grasps = sampler.generate_grasps(pc_ply, vis)
    print(f"Generated {len(grasps)} grasps.")

    # if vis:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o')
    #     for grasp in grasps:
    #         c1 = grasp['c1']['point']
    #         c2 = grasp['c2']['point']
    #         ax.scatter(c1[0], c1[1], c1[2], c='r', marker='x')
    #         ax.scatter(c2[0], c2[1], c2[2], c='g', marker='x')
    #         ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], c='k')
    #     plt.show()

def visualize_point_cloud(pcd):
    pc_ply = o3d.geometry.PointCloud()
    pc_ply.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pc_ply])

if __name__ == "__main__":
    point_cloud = np.load("/home/exx/Yubin/deep_mimic/mocap/utils/pcd_test.npy", allow_pickle=True)
    # visualize_normals(point_cloud)
    # test_antipodal_grasp_sampling(point_cloud, vis=True)

    pc_ply = o3d.geometry.PointCloud()
    pc_ply.points = o3d.utility.Vector3dVector(point_cloud)
    pc_ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=15))
    # pc_ply.orient_normals_consistent_tangent_plane(5)
    # camera_location = np.mean(np.asarray(pc_ply.points), axis=0) + np.array([0, 0, 10])
    # pc_ply.orient_normals_towards_camera_location(camera_location)
    # o3d.visualization.draw_geometries([pc_ply]) 
    
    test_antipodal_grasp_sampling(pc_ply, vis=True)

    # sampler = AntipodalGraspSampler()
    # pc_ply = sampler.shuffle_pc_ply(pc_ply)
    # o3d.visualization.draw_geometries([pc_ply])