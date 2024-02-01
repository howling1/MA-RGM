import os
import numpy as np
import open3d as o3d
import nibabel as nib
import shutil
import random

from pathlib import Path
from skimage.measure import marching_cubes
from tqdm import tqdm

def nifti2mesh(nifti_dir, target_dir, limit):
    count = 0
    
    for file in tqdm(os.listdir(str(Path(nifti_dir)))):
        if (limit <= count):
            break
            
        _path = str(os.path.join(str(nifti_dir), file).replace('\\', '/'))
        _filename = os.path.basename(_path).split(".")[0]
        if _filename == "":
            continue
            
        if _filename.split("-")[-1] != "seg":
            continue

        body_segment = nib.load(_path)
        body_segment_data = body_segment.get_fdata()
        verts, faces, _, __ = marching_cubes(body_segment_data, level=0, step_size=1)
        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(np.asarray(verts)),
                                            triangles=o3d.utility.Vector3iVector(np.asarray(faces)))

        _target_path =  target_dir + "/" + _filename + ".ply"
        o3d.io.write_triangle_mesh(_target_path, mesh)

        count += 1

def decimate_mesh(mesh_dir, target_dir, faces, limit):
    count = 0
    
    for file in tqdm(os.listdir(str(Path(mesh_dir)))):
        if (limit <= count):
            break
            
        _path = str(os.path.join(str(mesh_dir), file).replace('\\', '/'))
        _filename = os.path.basename(_path).split(".")[0]

        if _filename == "":
            continue

        mesh = o3d.io.read_triangle_mesh(_path)
        decimated_mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, faces)

        # delete nodes whose degree is 0
        vertices = np.asarray(decimated_mesh.vertices)
        triangles = np.asarray(decimated_mesh.triangles)
        vertex_degrees = np.zeros(len(vertices), dtype=int)
        for i in range(len(triangles)):
            v1, v2, v3 = triangles[i]
            vertex_degrees[v1] += 1
            vertex_degrees[v2] += 1
            vertex_degrees[v3] += 1

        zero_degree_vertices = np.where(vertex_degrees == 0)[0]
        decimated_mesh.remove_vertices_by_index(zero_degree_vertices)

        # save decimated mesh
        _target_path =  target_dir + "/" + _filename + ".ply"
        o3d.io.write_triangle_mesh( _target_path, decimated_mesh)
        
        count += 1
        
def mesh2pc(mesh_dir, target_dir, n_points, limit):
    count = 0
    
    for file in tqdm(os.listdir(str(Path(mesh_dir)))):
        if (limit <= count):
            break
            
        _path = str(os.path.join(str(mesh_dir), file).replace('\\', '/'))
        _filename = os.path.basename(_path).split(".")[0]

        if _filename == "":
            continue

        mesh = o3d.io.read_triangle_mesh(_path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=n_points)

        # # rescale the point cloud to a unit sphere and center it
        points = np.asarray(point_cloud.points)
        center = np.mean(points, axis=0)
        point_cloud.translate(-center)
        max_distance = np.max(np.linalg.norm(points - center, axis=1))
        point_cloud.scale(1.0/ max_distance, center=(0, 0, 0))

        # save pointcloud
        _target_path =  target_dir + "/" + _filename + ".ply"
        o3d.io.write_point_cloud(_target_path, point_cloud)

        count += 1

def data_split(pointcloud_dir, train_portion=0.8):

    train_dir = os.path.join(str(pointcloud_dir), "train")
    test_dir = os.path.join(str(pointcloud_dir), "test")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    file_list = [f for f in os.listdir(str(Path(pointcloud_dir))) if f.endswith(".ply")]
    random.shuffle(file_list)

    for i in tqdm(range(len(file_list))): 
        file = file_list[i]
        _path = str(os.path.join(str(pointcloud_dir), file).replace('\\', '/'))
        _filename = os.path.basename(_path).split(".")[0]

        if i <= len(file_list) * train_portion:
            shutil.move(_path, train_dir)
        else:
            shutil.move(_path, test_dir)

if __name__ == '__main__':
    nifti_dir = "/home/guests/siyu_zhou/dataset/ribseg_dataset"
    mesh_dir = "/home/guests/siyu_zhou/dataset/ribseg_mesh"
    decimated_mesh_dir = "/home/guests/siyu_zhou/dataset/ribseg_mesh_5k"
    pointcloud_dir = "/home/guests/siyu_zhou/dataset/ribseg_pointcloud_2048"

    nifti2mesh(nifti_dir, mesh_dir, limit = 1000)
    decimate_mesh(mesh_dir, decimated_mesh_dir, faces=5000, limit=1000)
    mesh2pc(mesh_dir, pointcloud_dir, n_points=2048, limit=1000)
    data_split(pointcloud_dir=pointcloud_dir)
