# plot spatial uncertainty of all experiments:
# for each model
# given different R values
# given 
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List
from tools import uncertainty_plot
from pytorch3d.ops import box3d_overlap

def post_process_models(basePath: str, setPrefix: str, modelPrefix: str, gtChangePoints: np.ndarray, modelSuffix: str = '.pth', device: str = 'cuda'):
        '''
        Iterates through all models generated during an experiment, and calculates their results.

        Args:
            - 
        '''

        # Function to check if a file exists
        # def model_exists(setNum, modelNum, basePath, setPrefix, modelPrefix, modelSuffix):
        #     set_path = os.path.join(basePath, f'{setPrefix}{setNum}', 'models')
        #     model_path = os.path.join(set_path, f'{modelPrefix}{modelNum}{modelSuffix}')
        #     return os.path.isfile(model_path)

        # List all directories in the base_path that start with 'set'
        set_dirs = [d for d in os.listdir(basePath) if os.path.isdir(os.path.join(basePath, d)) and d.startswith(setPrefix)]
        set_dirs.sort(key=lambda x: int(x[len(setPrefix):]))

        # logic for running through every model in every set (mutiple models in ensemble case)
        for set_name in set_dirs:
            model_num = 0
            while True:
                # if not model_exists(set_name, model_num, basePath, setPrefix, modelPrefix, modelSuffix):
                #     if model_num == 0:
                #         print(f'No models found in set {set_name}')
                #     break  # No more models in this set
                model_path = os.path.join(basePath, set_name, 'models', f'{modelPrefix}{model_num}{modelSuffix}')
                # model processing code
                change_model = torch.load(model_path).to(device)

                if set_name != 'set0':

                    print(f'Processing model {model_num} in set {set_name}: {model_path}')

                    best_result = post_process_model_i(groundtruthPoints=gtChangePoints, changedModel=change_model, # model for set 20. Need to make iterate across all sets
                                radii=[1, 1.5, 2, 2.5, 3, 3.5, 4],
                                densityThresholds=[0.5, 1, 2, 3, 4, 6, 8, 10, 20], # how many of the intial points to threshold out - (need some for CoV)
                                changeThresholds=[.1, .2, .3, .4, .5, .6, .7, .8, .9], # percentage of uncertainty range at which to threshold out points
                                resultDir=f"{base_path}{set_name}/figures",
                                device=device)
                    print(f"Best result for {set_name}: {best_result}")
                    model_num += 1 # used if there are ensembles

                break

def get_hyperparameter_ranges() -> List[List[int]]:
    '''
    Determines what range is available for testing of density and change
    '''


def post_process_model_i(groundtruthPoints: np.ndarray, changedModel: nn.Module, radii: List[int],
                          densityThresholds: List[int], changeThresholds: List[int], resultDir: str, device: str ='cuda'): # should be getting these values from the config, as lists
    '''
    Given a ground truth points, evaluate metrics across model parameters.

    args:
        - groundtruthPoints: the ground truth change points
        - changedModel: the nerf model trained on the mixed training data of scene 1 and scene 2
        - radii: list of desired radii you would like to test for neighbor algorithm
        - densityThresholds: List of densities you would like to evaluate at
        - changeThresholds: List of CoV's you would like to cut off points at (High CoV denotes change)
        - resultsDir: Where the results are saved to. format: path to directory. EX: experiments/test/set10/figures

        TODO: make it so post-process had a param for the location results.txt should be (make it figures for the set)
    '''
    results_path = f"{resultDir}/results.txt"
    best_result = [0,0,0,0,0]

    with open(results_path, 'a') as file:
        headers = ['radius', 'density threshold', 'change threshold', '3D IoU', 'Intersection Vol.']
        headers_str = ' '.join(headers)
        file.write(headers_str + '\n')

    for r in radii: # where we test multiple radius
        for thr in densityThresholds:
            for threshold in changeThresholds: # percentages

                estimated_change_pts = get_nerf_uncert_threshold_pts(model=changedModel, densityThreshold=thr, changeThreshold=threshold,
                                   device='cuda', N=100, neighborRadius=r, plotting=False)
                
                if len(estimated_change_pts) < 8: # if not enough points to form a 3d bbox then simply return none
                    with open(results_path, 'a') as file:
                        values = [r, thr, threshold, 'NONE', 'NONE']
                        values_str = ' '.join(map(str, values))
                        file.write(values_str + '\n')
                else:   
                    arr1 = groundtruthPoints # (M x 3) # make certain this is the right dims
                    arr2 = estimated_change_pts # (N x 3)

                    # Evaluation method 1
                    iou_3d, intersection_vol = get_nerfChange_boxIOU(torch.from_numpy(arr1).to(torch.float32), torch.from_numpy(arr2).to(torch.float32))

                    if iou_3d > best_result[3]:
                        best_result = [r,thr,threshold,iou_3d,intersection_vol]

                    with open(results_path, 'a') as file:
                        values = [r, thr, threshold, iou_3d, intersection_vol]
                        values_str = ' '.join(map(str, values))
                        file.write(values_str + '\n')

    return best_result
    


def get_nerfChange_boxIOU(changeGroundtruthPoints: torch.tensor,
                           changeEstimatePoints: torch.tensor ) -> Tuple[torch.tensor, torch.tensor]:
    '''
    Finds IOU between two bounding boxes. Uses pyTorch3D box3d_overlap function.
    '''
    box1 = get_bbox_3d(changeGroundtruthPoints)
    box2 = get_bbox_3d(changeEstimatePoints)

    try:
        intersection_vol, iou_3d = box3d_overlap(box1, box2)
    except ValueError as e:
         print(f"Caught an exception: {e}, returning iou and volume as -1")
         intersection_vol, iou_3d = -1, -1
    
    return iou_3d, intersection_vol


def get_bbox_3d(myPoints: torch.tensor) -> torch.tensor:
    '''
    Finds the 3d bbox given a pointcloud.

    args:
        points of shape (N,3)
        Note: Note:
        corners must be in a specific order to work with pytorch3d.ops.box3d_overlap

    returns:
        points defining the corners of the bounding box
    '''
    min_coords = torch.min(myPoints, dim=0).values
    max_coords = torch.max(myPoints, dim=0).values

    corners = torch.tensor([
    [max_coords[0], min_coords[1], max_coords[2]],
    [max_coords[0], max_coords[1], max_coords[2]],
    [max_coords[0], max_coords[1], min_coords[2]],
    [max_coords[0], min_coords[1], min_coords[2]],
    [min_coords[0], min_coords[1], max_coords[2]],
    [min_coords[0], max_coords[1], max_coords[2]],
    [min_coords[0], max_coords[1], min_coords[2]],
    [min_coords[0], min_coords[1], min_coords[2]],
    ])

    bounding_box_tensor = corners.unsqueeze(0)
    assert bounding_box_tensor.size() == (1,8,3)

    return bounding_box_tensor

# define what points are of the change class, and which are not
# def label_and_score(changePts: np.ndarray, estimatedChangePts: np.ndarray) -> float:
#     '''
#     Labels points as being of class 0 (change) or not.

#     Args:
#         changePts (ndarray): True set of changed points.
#         estimatedChangePts (ndarray): Estimated set of changed points.
#     Returns:
        
#     '''
#     changePts_set = set(map(tuple, changePts.T))

#     true_positives = []
#     false_positives = []

#     for pt in estimatedChangePts:
#         if tuple(pt) in changePts_set:
#             true_positives.append(pt)
#         else:
#             false_positives.append(pt)
    
#     true_positives = np.array(true_positives)
#     false_positives = np.array(false_positives)

#     true_labels = np.zeros(true_positives.shape[0])  # Class 0 for true positives
#     false_labels = np.ones(false_positives.shape[0]) 

#     #d = np.vstack((true_positives, false_positives))
#     y = np.hstack((true_labels, false_labels))

#     scores = np.hstack((np.ones(true_positives.shape[0]), np.zeros(false_positives.shape[0])))
#     roc_auc = roc_auc_score(y, scores)

#     return roc_auc


def get_nerf_uncert_threshold_pts(model: nn.Module, densityThreshold: int, changeThreshold: int, device: str = 'cuda',
                                   N: int = 100, neighborRadius: int= 2, plotting: bool = False) -> np.ndarray:
    '''
    returns points which are remaining after a given threshold, which we
    designate as change.

    args:
        - model: NeRF model with mixed data
        - densityThreshold: For thresholding out small ammount of low density points. Choose around 1.
        - neighborRadius: how close nearest neighbors should be considered. 2 is a good value.
    '''
    pts, density = get_nerf_pts(model, device, N) # ensure pts here is what is expected
    density = density.reshape(N, N, N)
    pts, values  = uncertainty_plot(scalar_field=density, scalars=None, pts=None,
                                     threshold=densityThreshold, plot=False, plotTitle="density plot")

    # Note: neighbor idx returns the index in the array of all pts of the neighbors.
    map_tree = KDTree(pts)
    neighbor_idx = map_tree.query_ball_point(x=pts, r=neighborRadius) # returns list of indices of neighbors of x
    
    CoV = [] # list of coefficent of variation for each point based on variation of neighbor densities
    for i in range(len(neighbor_idx)):
        neighbors = neighbor_idx[i] # [len(neighbors of that point)]

        surrounding_densities = [] # reinit list of neighbor for each point in the nerf
        for j in range(len(neighbors)):
            surrounding_densities.append(values[neighbors[j]]) # [n] densitity at each index in neighbors

        densities_tensor = torch.Tensor(surrounding_densities)
        mean_density = torch.mean(densities_tensor, dim=0) # calculates the arithmetic mean
        
        squared_diffs = []
        for d in densities_tensor: # for each point, subtract the mean density from each element.

            squared_diffs.append(torch.square(torch.sub(d, mean_density))) # squared difference for all points
        
        tensor_sum = torch.sum(torch.stack(squared_diffs), dim=0)
        divisor = len(neighbors) - 1
        sample_variance = torch.div(tensor_sum, divisor)
        coeff_variation = torch.div(torch.sqrt(sample_variance), mean_density)
        CoV.append(coeff_variation)

    # Find range of CoV and threshold based on it:
    uncert_array = np.stack([t.numpy() for t in CoV])
    max_uncertainty = np.nanmax(uncert_array)
    estimated_change_pts = uncertainty_plot(scalar_field=None, scalars=np.array(CoV), pts=pts,
                                             threshold=max_uncertainty*changeThreshold, plot=plotting, plotTitle="CoV plot")

    return estimated_change_pts
    

# Run across wide range of thresholds, evaluating the metric
def get_nerf_change_pts(k1Model: nn.Module, k2Model: nn.Module,
                         device: str = 'cuda', N: int = 100) -> np.ndarray:
    '''
    Gets the points of the change between two consecutive nerf models

    args:
        - k1_model: Loaded .pth file of the trained, original model
        - k2_model Loaded .pth file of the trained, changed model
    '''
    k1_pts = set(map(tuple, get_nerf_pts(k1Model, device, N)[0].T))# get points of original
    k2_pts = set(map(tuple, get_nerf_pts(k2Model, device, N)[0].T))# gets points of the changed nerf
    #print(f"shape of k1: {np.shape(k1_pts)}")

    #change_pts = k1_pts.symmetric_difference(k2_pts)
    change_pts = k1_pts - k2_pts
    change_pts = np.array(list(change_pts)).T

    return change_pts

    
def get_nerf_pts(model: nn.Module, device: str ='cuda', N: int = 100) -> np.ndarray:
    '''
    Gets the points of a nerf

    args:
        - neural networking model

    returns:
        -  points: 3xN array
        - density: (NxNxN) array
    '''
    scale = 1.5

    x = torch.linspace(-scale, scale, N)
    y = torch.linspace(-scale, scale, N)
    z = torch.linspace(-scale, scale, N)

    x, y, z = torch.meshgrid((x, y, z))

    xyz = torch.cat((x.reshape(-1, 1),
                    y.reshape(-1, 1),
                    z.reshape(-1, 1)), dim=1)
    
    with torch.no_grad():
        _, density = model.forward(xyz.to(device), torch.zeros_like(xyz).to(device))
    
    density = density.cpu().numpy().reshape(N, N, N)

    threshold = 1 # threshold gets rid of some noise points

    above_threshold_mask = density > threshold
    x_coords, y_coords, z_coords = np.where(above_threshold_mask)
    # np.save('mask.npy', (x_coords, y_coords, z_coords))
    pts = np.array([x_coords, y_coords, z_coords])

    return pts, density


if __name__ == '__main__':
    print("testing functions!")
    device = 'cuda'
    #pth_file_k1 = 'experiments/test/set0/models/M0.pth'
    #pth_file_k2 = 'experiments/pigeye/models/M0.pth'
    #model_k1 = torch.load(pth_file_k1).to(device)
    #model_k2 = torch.load(pth_file_k2).to(device)
    change_model = torch.load('experiments/test/set20/models/M0.pth').to(device)

    import os
    # new stuff for testing
    base_path = 'experiments/test/'
    set_prefix = 'set'
    model_prefix = 'M'

    pcd = o3d.io.read_point_cloud("gt_test_change.pcd")
    gt_points = np.asarray(pcd.points)

    # estimated_change_pts = get_nerf_uncert_threshold_pts(model=change_model, densityThreshold=17.0, changeThreshold=0, # why is it not getting anything if less than 7.7
    #                                device='cuda', N=100, neighborRadius=1, plotting=False)
    
    post_process_models(basePath=base_path, setPrefix=set_prefix, modelPrefix=model_prefix,
                         gtChangePoints=gt_points, modelSuffix='.pth', device='cuda')
    


#else: # something elses
    # device = 'cuda'
    # pth_file_k1 = 'experiments/test/set0/models/M0.pth'
    # pth_file_k2 = 'experiments/pigeye/models/M0.pth'
    # model_k1 = torch.load(pth_file_k1).to(device)
    # model_k2 = torch.load(pth_file_k2).to(device)
    # change_model = torch.load('experiments/test/set20/models/M0.pth').to(device)
    # change_pts_gt = get_nerf_change_pts(k1Model=model_k1, k2Model=model_k2, device='cuda')

    # #label_and_score(changePts=change_pts, estimatedChangePts)
    # #uncertainty_plot(scalar_field=None, scalars=np.ones(len(change_pts_gt.T)), pts=change_pts_gt.T, threshold=0)
    # estimated_change_pts = get_nerf_uncert_threshold_pts(model=change_model, densityThreshold=10, changeThreshold=100,
    #                                device='cuda', N=100, neighborRadius=3, plotting=True)
    
    # arr1 = change_pts_gt.T # (M x 3)
    # arr2 = estimated_change_pts # (N x 3)

    # iou_3d, intersection_vol = get_nerfChange_boxIOU(torch.from_numpy(arr1).to(torch.float32), torch.from_numpy(arr2).to(torch.float32))
    # print(f"the iou is: {iou_3d}")
    # print(f"the intersection volume is: {intersection_vol}")


    # box1 = get_bbox_3d(torch.from_numpy(arr1).to(torch.float32))
    # box2 = get_bbox_3d(torch.from_numpy(arr1).to(torch.float32))
    # intersection_vol, iou_3d = box3d_overlap(box1, box2)
    # print(f"the iou is: {iou_3d}")
    # print(f"the intersection volume is: {intersection_vol}")
    
    # points1_np = box1.numpy().squeeze()
    # points2_np = box2.numpy().squeeze()
    # # Extract coordinates
    # x1, y1, z1 = points1_np[:, 0], points1_np[:, 1], points1_np[:, 2]
    # x2, y2, z2 = points2_np[:, 0], points2_np[:, 1], points2_np[:, 2]
    # # Plot the points in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x1, y1, z1, c='r', marker='o', label='Points Set 1')
    # ax.scatter(x2, y2, z2, c='b', marker='^', label='Points Set 2')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.legend()
    # plt.show()

    # # dtype = [('x', arr1.dtype), ('y', arr1.dtype), ('z', arr1.dtype)]
    # # arr1_structured = np.array([tuple(row) for row in arr1], dtype=dtype)
    # # arr2_structured = np.array([tuple(row) for row in arr2], dtype=dtype)

    # # common = np.intersect1d(arr1_structured, arr2_structured)
    # # print(f"Number of common elements: {len(common)}")
    # # common = np.array([[point['x'], point['y'], point['z']] for point in common]).T
    
    # # test_gt = np.array([
    # #     [1, 2, 3],
    # #     [4, 5, 6],
    # #     [7, 8, 9],
    # #     [10, 11, 12],
    # #     [13, 14, 15],
    # #     [16, 17, 18],
    # #     [19, 20, 21],
    # #     [22, 23, 24],
    # #     [25, 26, 27],
    # #     [28, 29, 30]
    # # ])

    # # test_est = np.array([
    # #     [1, 2, 3],
    # #     [4, 5, 6],
    # #     [7, 8, 9],
    # #     [10, 11, 12],
    # #     [13, 14, 15],
    # #     [31, 32, 33],
    # #     [34, 35, 36],
    # #     [37, 38, 39],
    # #     [40, 41, 42],
    # #     [43, 44, 45]
    # # ])

    # # # Combine and find unique points from both arrays
    # # all_points = np.vstack((test_gt, test_est))
    # # all_points_unique = np.unique(all_points, axis=0)

    # # gt_labels = np.array([1 if point in test_gt.tolist() else 0 for point in all_points_unique.tolist()])
    # # est_labels = np.array([1 if point in test_est.tolist() else 0 for point in all_points_unique.tolist()])
    # # # just because it isn't included in est_labels doesn't mean it mislabeled it
    # # # Compute the ROC AUC score
    # # roc_auc = roc_auc_score(gt_labels, est_labels)
    # # print(f"ROC AUC Score: {roc_auc}")
    # # conf_matrix = confusion_matrix(gt_labels, est_labels)
    # # # create the axis to plot onto 
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111)

    # # # plot the matrix
    # # cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    # # fig.colorbar(cax)

    # # # labels 
    # # plt.xlabel('Predicted')
    # # plt.ylabel('Expected')

    # # plt.show()

    # #print(label_and_score(changePts=test_gt, estimatedChangePts=test_est))
    # #print(label_and_score(changePts=change_pts_gt, estimatedChangePts=estimated_change_pts))

    # print( "function testing complete")