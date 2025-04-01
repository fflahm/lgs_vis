import numpy as np
import torch
import open3d as o3d
import sys

sys.path.append("..")
from scene.gaussian_model import GaussianModel
from autoencoder.model import Autoencoder

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import random

from o3d_utils import load_camera, visualize_point_cloud

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def draw_hist(data, label):
    plt.hist(data, bins=30, density=False, alpha=0.6, color='b', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.title(f'Histogram of {label}')
    plt.show()

if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("-m", type=str, default="../data/lerf/teatime/output/teatime_3/chkpnt30000.pth")
    parser.add_argument("--decoding", action="store_true")
    parser.add_argument("--ae_ckpt", type=str, default="../data/lerf/teatime/ae_ckpt/best_ckpt.pth")
    parser.add_argument("--load_cam", action="store_true")
    parser.add_argument("--cam_id", type=int, default=25)
    parser.add_argument("--cam_json", type=str, default="../data/lerf/teatime/output/teatime_3/cameras.json")
    parser.add_argument("--draw_hist", action="store_true")
    parser.add_argument("--mult_by_opac", action="store_true")
    parser.add_argument("--opac_filter", action="store_true")
    parser.add_argument("--opac_thresh", type=float, default=0.2)
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--outlier_filter", action="store_true")
    parser.add_argument("--outlier_thresh", type=float, default=3.0)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    args.pca = args.pca | args.decoding # features after decoding need pca
    np.random.seed(args.seed)       
    random.seed(args.seed)          
    torch.manual_seed(args.seed)

    # Load gaussian model
    gaussians = GaussianModel(3)
    (model_params, _) = torch.load(args.m)
    gaussians.restore(model_params, None, mode="test")
    points = gaussians._xyz.detach().cpu().numpy() # (N, 3)
    colors = gaussians._language_feature.detach().cpu() # (N, 3)
    opacity = gaussians._opacity.squeeze(1).detach().cpu().numpy() # (N,)
    opacity = sigmoid(opacity) # most from 0 to 1
    if args.draw_hist:
        for i in range(3):
            draw_hist(colors[:, i], f"Color_{i}_Origin")
        draw_hist(opacity, "Opacity")

    # In my opinion, 3D language features may need to be multiplied by opacity
    if args.mult_by_opac:
        colors = colors * opacity[:, None]
        if args.draw_hist:
            for i in range(3):
                draw_hist(colors[:, i], f"Color_{i}_Multiple")

    # Opacity filter
    if args.opac_filter:
        filtered_indices = opacity >= args.opac_thresh
        points = points[filtered_indices] # (n, 3)
        colors = colors[filtered_indices] # (n, 3)
        print(f"Total: {len(opacity)}, Remaining: {len(points)}") # Total N, Remaining n
        if args.draw_hist:
            for i in range(3):
                draw_hist(colors[:, i], f"Color_{i}_Filtered")

    # Decoding
    if args.decoding:
        checkpoint = torch.load(args.ae_ckpt)
        model = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512])
        model.load_state_dict(checkpoint)
        model.eval()
        colors = model.decode(colors) # (n, 512)

    # PCA
    if args.pca:
        # colors = (colors - colors.mean(dim=0, keepdim=True)) / colors.std(dim=0, keepdim=True)
        _, _, v = torch.pca_lowrank(colors)
        colors = torch.matmul(colors, v[..., :3]).detach()
        if args.draw_hist:
            for i in range(3):
                draw_hist(colors[:, i], f"Color_{i}_PCA")

    # Outlier filter
    if args.outlier_filter:
        d = torch.abs(colors - torch.median(colors, dim=0).values)
        mdev = torch.median(d, dim=0).values
        s = d / mdev
        m = args.outlier_thresh  # this is a hyperparam controlling how many std dev outside for outliers
        rins = colors[s[:, 0] < m, 0]
        gins = colors[s[:, 1] < m, 1]
        bins = colors[s[:, 2] < m, 2]
        print(f"Total: {len(colors)}, Remaining: R {len(rins)}, G {len(gins)}, B {len(bins)}")
    else:
        rins = colors[:, 0]
        gins = colors[:, 1]
        bins = colors[:, 2]

    # Normalize
    colors[:, 0] -= rins.min()
    colors[:, 1] -= gins.min()
    colors[:, 2] -= bins.min()
    colors[:, 0] /= rins.max() - rins.min()
    colors[:, 1] /= gins.max() - gins.min()
    colors[:, 2] /= bins.max() - bins.min()
    colors = torch.clamp(colors, 0, 1).numpy()
    if args.draw_hist: 
        for i in range(3):
            draw_hist(colors[:, i], f"Color_{i}_Normalized")

    # Visualize
    if args.load_cam:
        visualize_point_cloud(points, colors, load_camera(args.cam_json, args.cam_id), save_path=args.save_path)
    else:
        visualize_point_cloud(points, colors, save_path=args.save_path)
