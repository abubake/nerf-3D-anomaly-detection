import torch
import numpy as np
import pyvista as pv
import open3d as o3d
from evaluate import get_nerf_pts
from evaluate import get_nerf_uncert_threshold_pts
from evaluate import get_bbox_3d
import tools as tools

def plot_object_with_bbox(changePoints: np.ndarray, objectPoints: np.ndarray,
                           plotTitle: str = "Object with bbox estimate", gtChangePoints: np.ndarray = None):
    '''
    Plots the bounding box result on a pyvista plot

    args:
        - changePoints: points representing the 3D change (N, 3)
        - objectPoints: points representing the object (N, 3)
        - gtChangePoints: Optional np.ndarray of shape (M, 3) representing the 3D coordinates of the ground truth change points. 
                          A green bounding box will be drawn if provided.
    '''
    changePoints_tensor = torch.tensor(changePoints)
    bbox = get_bbox_3d(changePoints_tensor) # requires tensor
    bbox_np = bbox.detach().cpu().numpy().reshape(-1, 3)

    plotter = pv.Plotter(title=plotTitle)
    plotter.add_points(objectPoints, scalars=np.ones(len(objectPoints)), cmap="inferno")
    plotter.add_points(changePoints, scalars=np.ones(len(changePoints)), color='blue')
    plotter.add_points(bbox_np, color='red')

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Bottom face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical lines connecting top and bottom faces
    ]
    # Draw red lines between the corners of the bounding box
    for edge in edges:
        line = np.array([bbox_np[edge[0]], bbox_np[edge[1]]])
        plotter.add_lines(line, color='red', width=2)

    if gtChangePoints is not None:
        gtChangePoints_tensor = torch.tensor(gtChangePoints)
        gt_bbox = get_bbox_3d(gtChangePoints_tensor)
        gt_bbox_np = gt_bbox.detach().cpu().numpy().reshape(-1, 3)
        
        # Add the ground truth bounding box points
        plotter.add_points(gt_bbox_np, color='green')

        # Draw green lines between the corners of the ground truth bounding box
        for edge in edges:
            line = np.array([gt_bbox_np[edge[0]], gt_bbox_np[edge[1]]])
            plotter.add_lines(line, color='green', width=2)

    plotter.show(auto_close=True)


def pyvista_plot(*points: np.ndarray, values: np.ndarray, plotTitle: str):
    '''
    Plots.

    args:
        - points: array of either all points (Nx3) or each coordinate of points (Nx1), (Nx1), (Nx1)
        - values: value at each point (array). can be passed as np.ones_like(len(points))
        - plotTitle: Title of plot.
    '''
    plotter = pv.Plotter(title=plotTitle)

    if len(points) not in (1, 3):
        print("Points must either be given as one variable, or as three.")
        exit
    elif len(points) == 1:
        plotter.add_points(points[0], scalars=values, cmap="inferno")
        plotter.show(auto_close=True)
    else: # when 3
        plotter.add_points(np.column_stack((points[0], points[1], points[2])), scalars=values, cmap="inferno")
        plotter.show(auto_close=True)



if __name__ == '__main__':

    device = 'cuda'
    pth_file_k1 = 'experiments/whale/set0/models/M0.pth'
    #pth_file_k2 = 'experiments/pigeye/models/M0.pth'
    model_k1 = torch.load(pth_file_k1).to(device)
    #model_k2 = torch.load(pth_file_k2).to(device)
    change_model = torch.load('experiments/whale/set25/models/M0.pth').to(device) # change whale to test or other model name

    # change_pts_gt = get_nerf_change_pts(k1Model=model_k1, k2Model=model_k2, device='cuda') # (3xN)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(change_pts_gt.T)
    # o3d.io.write_point_cloud("gt_change.pcd", pcd)

    pcd = o3d.io.read_point_cloud("data/gt_change/gt_whale_change.pcd")
    gt_points = np.asarray(pcd.points)

    #pyvista_plot(gt_points, values=np.ones(len(gt_points)), plotTitle="ground truth change")
    
    estimated_change_pts = get_nerf_uncert_threshold_pts(model=change_model, densityThreshold=1, changeThreshold=.9,
                                      device='cuda', N=100, neighborRadius=2, plotting=False) # (Nx3)
    
    nerf_points, density = get_nerf_pts(model=model_k1)
    N = 100
    densityThreshold = 7 # was 7
    density = density.reshape(N, N, N)
    pts, values  = tools.uncertainty_plot(scalar_field=density, scalars=None, pts=None,
                                     threshold=densityThreshold, plot=False, plotTitle="density plot")

    plot_object_with_bbox(changePoints=estimated_change_pts, objectPoints=pts,
                           plotTitle="Best box so far", gtChangePoints=gt_points)
    
    # pyvista_plot(estimated_change_pts, values=np.ones(len(estimated_change_pts)), plotTitle="est. change")

    # Visualize gt points