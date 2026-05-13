import torch
import os
import sys
import h5py
sys.path.append(os.getcwd())
from graph.build_graph import *
from graph.error_metrics import *


# TODO check why gt error is so huge

## load data
DATA_PATH = os.path.join(os.getcwd(), "../data", "freehand_us", "scan_h5_files")

# inbetween_transforms_pred = torch.load(BASE_PATH + '/pose_data/inbetween_transforms_pred.pt') # (N, 4, 4)
# acc_transforms_pred = torch.load(BASE_PATH + '/pose_data/acc_transforms_pred.pt') # (N, 4, 4)

# inbetween_transforms_gt = torch.load(BASE_PATH + '/pose_data/inbetween_transforms_gt.pt') # (N, 4, 4)
# acc_transforms_gt = torch.load(BASE_PATH + '/pose_data/acc_transforms_gt.pt') # (N, 4, 4)

# BASE_PATH = os.path.join(os.getcwd(), "../freehand_adapted", "results", "seq_len10__lr0.0001__pred_type_parameter__label_type_point")

with h5py.File(DATA_PATH + '/scan_0000.h5', 'r') as f:
    inbetween_transforms_pred = torch.from_numpy(f['pred/inbetween_transforms_pred'][:]) # (N, 4, 4)
    acc_transforms_pred = torch.from_numpy(f['pred/acc_transforms_pred'][:]) # (N, 4, 4)
    inbetween_transforms_gt = torch.from_numpy(f['gt/inbetween_transforms_gt'][:]) # (N, 4, 4)
    acc_transforms_gt = torch.from_numpy(f['gt/acc_transforms_gt'][:]) # (N, 4, 4)

# remove initial zero element and last element
inbetween_transforms_pred = inbetween_transforms_pred[1:-1]
acc_transforms_pred = acc_transforms_pred[1:-1]
inbetween_transforms_gt = inbetween_transforms_gt[1:-1]
acc_transforms_gt = acc_transforms_gt[1:-1]

# # build graphs
graph_pred, initial_pred, optimized_pred = build_graph(acc_transforms_pred, inbetween_transforms_pred, True)
graph_gt, initial_gt, optimized_gt = build_graph(acc_transforms_gt, inbetween_transforms_gt, True)

# plot marginals
#plot_marginals(graph_pred, optimized_pred, 1, 10)

# plot trajectories (no rotation!)
# plot_trajectories([extract_positions(acc_transforms_pred, pose_type="torch_tensor"), extract_positions(acc_transforms_gt, pose_type="torch_tensor")],
#                   labels=["Initial estimated", "GT"],
#                   colors=["blue", "red"])

# plot_trajectories([extract_positions(initial_pred, pose_type="gtsam_values"), extract_positions(optimized_pred, pose_type="gtsam_values")],
#                   labels=["pred initial", "pred acc"],
#                   colors=["blue", "red"])

# error metrics
error_pred_initial = graph_pred.error(initial_pred)
error_pred_optimized = graph_pred.error(optimized_pred)
error_gt_initial = graph_gt.error(initial_gt)
error_gt_optimized = graph_gt.error(optimized_gt)

print("\n\n==== Errors ====\n")

print(f"Initial error pred: {error_pred_initial}\n")
print(f"Optimized error pred: {error_pred_optimized}\n")
print(f"Initial error gt: {error_gt_initial}\n")
print(f"Optimized error gt: {error_gt_optimized}\n")

avg_t_err, avg_r_err = avg_trajectory_error(acc_transforms_gt, acc_transforms_pred)
print(f"Average translation error acc\n: {avg_t_err}\n")
print(f"Average rotation error acc\n: {avg_r_err}\n")

avg_t_err, avg_r_err = avg_trajectory_error(inbetween_transforms_gt, inbetween_transforms_pred)
print(f"Average translation error inbetween\n: {avg_t_err}\n")
print(f"Average rotation error inbetween\n: {avg_r_err}\n")