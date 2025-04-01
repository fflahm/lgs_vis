import numpy as np
import torch
import matplotlib
from o3d_utils import visualize_point_cloud, load_camera
from scene.gaussian_model import GaussianModel
from autoencoder.model import Autoencoder
import sys
sys.path.append("eval")
from eval.openclip_encoder import OpenCLIPNetwork

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

# Init
gaussian_ckpt_path = (
    "data/lerf/teatime/output/teatime_3/chkpnt30000.pth"
    # "data/sofa/sofa_3/chkpnt30000.pth"
    # "data/640480/chkpnt30000.pth"
)
ae_ckpt_path = "data/lerf/teatime/ae_ckpt/best_ckpt.pth"
camera_json_path = "data/lerf/teatime/output/teatime_3/cameras.json"
device = "cuda:0"
query_text = "kuma"
cam_id = 25

# Load gaussian model
gaussians = GaussianModel(3)

(model_params, _) = torch.load(gaussian_ckpt_path)
gaussians.restore(model_params, None, mode="test")
points = gaussians._xyz.detach().cpu().numpy() # (N, 3)
feats = gaussians._language_feature.detach().to(device) # (N, 3)
opacity = gaussians._opacity.squeeze(1).detach().cpu().numpy() # (N,)
opacity = sigmoid(opacity) # most from 0 to 1

# Filter by opacity
print(f"Num before opacity filter: {len(points)}")
indices_opacity = opacity >= 0.9
points = points[indices_opacity]
feats = feats[indices_opacity]
print(f"Num after opacity filter: {len(points)}")

# Decode
checkpoint = torch.load(ae_ckpt_path)
model = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512])
model.load_state_dict(checkpoint)
model.eval()
model.to(device)
with torch.no_grad():
    feats = model.decode(feats) # [N, 512]

# Query
clip_model = OpenCLIPNetwork(device)
clip_model.set_positives([query_text])
scores = clip_model.get_max_across(feats.unsqueeze(0).unsqueeze(0)).squeeze().cpu().numpy() # [N,] scores

# Filter by score
print(f"Num before score filter: {len(points)}")
indices_score = scores >= 0.5
points = points[indices_score]
scores = scores[indices_score]
print(f"Num after score filter: {len(points)}")

# Normalize
max_score = scores.max()
min_score = scores.min()
scores = (scores - min_score) / (max_score - min_score) * 255
scores = np.clip(scores, 0, 255).astype(np.uint8)

# Visualize
colormap = np.array(matplotlib.colormaps["turbo"].colors)
colors = colormap[scores]
visualize_point_cloud(points, colors, 
        load_camera(camera_json_path, 2))