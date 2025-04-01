import numpy as np
import open3d as o3d
import json

def load_camera(json_path, id):
    with open(json_path, "r") as file:
        json_cams = json.load(file)
    return next((item for item in json_cams if item["id"] == id), None)

def visualize_point_cloud(points, colors, camera=None, indice=None, save_path=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if camera:
        W = camera["width"]
        H = camera["height"]
        fx = camera["fx"]
        fy = camera["fy"]
        intrinsic = o3d.pybind.camera.PinholeCameraIntrinsic(width=W, height=H, 
                        fx=fx, fy=fy, cx=(W-1) / 2,cy=(H-1) / 2)
        position = np.array(camera["position"])
        rotation = np.array(camera["rotation"])
        c2w = np.eye(4)
        c2w[:3, :3] = rotation  
        c2w[:3, 3] = position
        extrinsic= np.linalg.inv(c2w)

        param = o3d.pybind.camera.PinholeCameraParameters()
        param.extrinsic = extrinsic
        param.intrinsic = intrinsic

    vis = o3d.visualization.Visualizer()

    if camera:
        vis.create_window(width=W, height=H)
    else:
        vis.create_window()

    vis.add_geometry(pcd)
    if indice:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.paint_uniform_color([1, 0, 0])
        sphere.translate(points[indice])
        vis.add_geometry(sphere)

    if camera:
        view_control = vis.get_view_control()
        view_control.convert_from_pinhole_camera_parameters(param)

    vis.run()
    if save_path:
        vis.capture_screen_image(save_path)
    vis.destroy_window()