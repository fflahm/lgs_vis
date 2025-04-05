import numpy as np
import torch
import matplotlib
import random
from sklearn.cluster import KMeans
from scipy.special import softmax
from argparse import ArgumentParser
from o3d_utils import visualize_point_cloud, load_camera, draw_hist

import sys
sys.path.append("..")
from scene.gaussian_model import GaussianModel
from autoencoder.model import Autoencoder
from my_openclip import OpenCLIPNetwork

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--query", type=str, default="bear")
    parser.add_argument("-m", type=str, default="../data/lerf/teatime/output/teatime_3/chkpnt30000.pth")
    parser.add_argument("--ae_ckpt", type=str, default="../data/lerf/teatime/ae_ckpt/best_ckpt.pth")
    parser.add_argument("--load_cam", action="store_true")
    parser.add_argument("--cam_id", type=int, default=25)
    parser.add_argument("--cam_json", type=str, default="../data/lerf/teatime/output/teatime_3/cameras.json")
    parser.add_argument("--mult_by_opac", action="store_true")
    parser.add_argument("--opac_filter", action="store_true")
    parser.add_argument("--opac_thresh", type=float, default=0.2)
    parser.add_argument("--score_filter_up", action="store_true")
    parser.add_argument("--score_thresh_up", type=float, default=2.5)
    parser.add_argument("--score_filter_down", action="store_true")
    parser.add_argument("--score_thresh_down", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    np.random.seed(args.seed)       
    random.seed(args.seed)          
    torch.manual_seed(args.seed)

    # Load gaussian model
    gaussians = GaussianModel(3)
    (model_params, _) = torch.load(args.m)
    gaussians.restore(model_params, None, mode="test")
    points = gaussians._xyz.detach().cpu().numpy() # (N, 3)
    feats = gaussians._language_feature.detach().to(args.device) # (N, 3)
    opacity = gaussians._opacity.squeeze(1).detach().cpu().numpy() # (N,)
    opacity = sigmoid(opacity)
    if args.mult_by_opac:
        feats = feats * torch.tensor(opacity, device=args.device).unsqueeze(-1)

    # Filter by opacity
    if args.opac_filter:
        print(f"Num before opacity filter: {len(points)}")
        indices_opacity = opacity >= args.opac_thresh
        points = points[indices_opacity]
        feats = feats[indices_opacity]
        print(f"Num after opacity filter: {len(points)}")

    # Decode
    checkpoint = torch.load(args.ae_ckpt)
    model = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512])
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(args.device)
    # with torch.no_grad():
    #     feats = model.decode(feats) # (N, 512)

    # Query
    # clip_model = OpenCLIPNetwork(args.device, 
    #     ("wooden table", "ceiling", "wall", "window"))
    clip_model = OpenCLIPNetwork(args.device)
    # clip_model = OpenCLIPNetwork(args.device, 
    #         ("clock", "table", "floor", "door", "lamp", "wall", "window", "sofa"))
    # texts = ["clock", "table", "door", "wall", "window", "sofa"]
    clip_model.set_positives(["electric fan"])
    text_embed = clip_model.pos_embeds.to(feats.dtype)
    with torch.no_grad():
        text_embed = model.encode(text_embed)
    scores = torch.mm(feats, text_embed.T).squeeze().cpu().numpy() # (N,)
    
    # scores = clip_model.get_max_across(feats.unsqueeze(0).unsqueeze(0)).squeeze().cpu().numpy() # (N,) scores
    # scores = torch.mm(feats, clip_model.pos_embeds.T.to(feats.dtype)).squeeze().cpu().numpy()
    # scores = softmax(scores)[:, 0] # (N,)
    
    # labels = scores.argmax(axis=1)
    # draw_hist(scores, "Score_Init")
    # for i in range(len(texts)):
    #     print(texts[i], np.sum(labels == i))
    #     colors = np.ones_like(points)
    #     colors[labels == i, 1] = 0.0
    #     colors[labels == i, 2] = 0.0
    #     visualize_point_cloud(points, colors, 
    #             load_camera(args.cam_json, args.cam_id))


    # Filter by score
    # kmeans = KMeans(n_clusters=10, n_init=10, random_state=args.seed)
    # labels = kmeans.fit_predict(scores.reshape(-1, 1))
    # cluster_centers = kmeans.cluster_centers_.flatten()
    # scores = np.array([cluster_centers[label] for label in labels])
    # draw_hist(scores, "Score_Clustered")
    # groups = np.argsort(cluster_centers)[::-1]
    # for gid in groups:
    #     print(cluster_centers[gid])
    #     gind = labels == gid
    #     gc = np.zeros_like(points)
    #     gc[gind, 0] = 1.0
    #     visualize_point_cloud(points, gc, 
    #             load_camera(args.cam_json, args.cam_id))        
    mean = np.mean(scores)
    std = np.std(scores)
    if args.score_filter_up:
        print(f"Num before score filter up: {len(points)}")
        threshold = mean + std * args.score_thresh_up
        draw_hist(scores, "Score_Filtered_Up", threshold)
        indices_score = scores <= threshold
        points = points[indices_score]
        scores = scores[indices_score]
        print(f"Num after score filter up: {len(points)}")
    if args.score_filter_down:
        print(f"Num before score filter down: {len(points)}")
        threshold = mean + std * args.score_thresh_down
        draw_hist(scores, "Score_Filtered_Down", threshold)
        indices_score = scores >= threshold
        points = points[indices_score]
        scores = scores[indices_score]
        print(f"Num after score filter down: {len(points)}")

    # Normalize
    max_score = scores.max()
    min_score = scores.min()
    topk_values, topk_indices = torch.topk(torch.tensor(scores), 10)
    print(max_score, topk_values.numpy())
    topk_indices = topk_indices.numpy()
    scores = (scores - min_score) / (max_score - min_score) * 255
    # scores = np.clip(scores, 0, 255).astype(np.uint8)
    assert scores.max() <= 255
    assert scores.min() >= 0
    scores = scores.astype(np.uint8)

    # Visualize
    colormap = np.array(matplotlib.colormaps["turbo"].colors)
    colors = colormap[scores]
    visualize_point_cloud(points, colors, 
            load_camera(args.cam_json, args.cam_id), topk_indices, args.save_path)