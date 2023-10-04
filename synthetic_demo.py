import os
from archs.diffusion_extractor import DiffusionExtractor
from archs.aggregation_network import AggregationNetwork
from archs.stable_diffusion.diffusion import latent_to_image
from archs.stable_diffusion.resnet import collect_dims
from archs.correspondence_utils import (
  batch_cosine_sim,
  rescale_points,
  find_nn_correspondences,
  draw_correspondences,
  points_to_patches,
  compute_pck
)
import os
import random
import torch
import math
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import pandas as pd

# Memory requirement is 13731MiB
device = "cuda"
config_path = "configs/synthetic.yaml"
config = OmegaConf.load(config_path)
config = OmegaConf.to_container(config, resolve=True)

# dims is the channel dim for each layer (12 dims for Layers 1-12)
# idxs is the (block, sub-block) index for each layer (12 idxs for Layers 1-12)
diffusion_extractor = DiffusionExtractor(config, device)
dims = collect_dims(diffusion_extractor.unet, idxs=diffusion_extractor.idxs)

aggregation_network = AggregationNetwork(
  projection_dim=config["projection_dim"],
  feature_dims=dims,
  device=device,
  save_timestep=config["save_timestep"],
  num_timesteps=config["num_timesteps"]
)
aggregation_network.load_state_dict(torch.load(config["weights_path"], map_location="cpu")["aggregation_network"])

guidance_scale = 7.5
prompt = "A raccoon playing chess with oversized pieces."
negative_prompt = config["negative_prompt"]

diffusion_extractor.change_cond(prompt, "cond")
diffusion_extractor.change_cond(negative_prompt, "uncond")

latents = torch.randn((diffusion_extractor.batch_size, diffusion_extractor.unet.in_channels, 512 // 8, 512 // 8), device=diffusion_extractor.device, generator=diffusion_extractor.generator)
hash_name = str(random.getrandbits(32))

print("Guidance Scale:", guidance_scale)
print("Prompt:", diffusion_extractor.prompt)
print("Negative Prompt:", diffusion_extractor.negative_prompt)

with torch.inference_mode():
  feats, outputs = diffusion_extractor.forward(latents=latents, guidance_scale=guidance_scale)
  if feats is not None:
    """
    feats is the cached feature maps of shape (B, S, L, W, H) where
    B is the batch size
    S is the length of save_timestep
    L is the length of dims / the number of layers
    W, H is the dimension of the latent

    outputs is the x0 predictions (not xt) of length num_timesteps + 1 where
    the first element is the starting noise and the last is the final image
    """
    # reverse feats along the time dimension S
    # when passing to the aggregation network since it was trained
    # on feature maps from the inversion process (image to noise)
    # and we are running the generation process (noise to image)
    b, s, l, w, h = feats.shape
    diffusion_hyperfeats = aggregation_network(torch.flip(feats, dims=(1,)).float().view(b, -1, w, h))
synthetic_images = diffusion_extractor.latents_to_images(outputs[-1])

plt.clf()
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(synthetic_images[0])
axs[0].axis("off")
axs[1].imshow(synthetic_images[1])
axs[1].axis("off")
fig.suptitle(prompt)
fig.tight_layout()
plt.show()

# To interactively see what Diffusion Hyperfeatures predicts
# as the corresponding keypoint, simply set this flag to true
# and the prediction will appear as a green point on the right image.
# Your annotations will show up as red points in both images.
show_dhf = False





def get_corresponding_points(img1_feats, img2_feats, output_size, load_size):
  """
  Precompute nearest neighbor for every pixel in img1.
  To reduce memory usage, we do this computation in
  with descriptor maps of shape output_size and rescale to the load_size.
  """
  sims = batch_cosine_sim(img1_feats[None, ...], img2_feats[None, ...])
  num_pixels = int(math.sqrt(sims.shape[-1]))
  points1, points2 = find_nn_correspondences(sims)
  points1, points2 = points1[0].detach().cpu().numpy(), points2[0].detach().cpu().numpy()
  points1 = rescale_points(points1, output_size, load_size)
  points2 = rescale_points(points2, output_size, load_size)
  return points1, points2

# Set the load_size to 224 so that the PCK metric is comparable to prior work
# To view the predictions in full resolution, change load_size to (512, 512)
i, j = 0, 1
output_size = (config["output_resolution"], config["output_resolution"])
load_size = (224, 224)
img1, img2 = synthetic_images[i].resize(load_size), synthetic_images[j].resize(load_size)
points1_dhf, points2_dhf = get_corresponding_points(
  diffusion_hyperfeats[i],
  diffusion_hyperfeats[j],
  output_size,
  load_size
)

import matplotlib.pyplot as plt
import numpy as np

user_source_points, user_target_points = [], []
dhf_points = []

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img1)
axes[0].axis("off")
axes[1].imshow(img2)
axes[1].axis("off")
plt.draw()


def mouse_event(event):
  radius1, radius2 = 8, 1
  y, x = event.ydata, event.xdata

  if event.inaxes == axes[0]:
    color = "r"
    circ1_1 = plt.Circle((x, y), radius1, facecolor=color, edgecolor='white', alpha=0.5)
    circ1_2 = plt.Circle((x, y), radius2, facecolor=color, edgecolor='white')
    axes[0].add_patch(circ1_1)
    axes[0].add_patch(circ1_2)
    user_source_points.append((y, x))

    y_patch, x_patch = points_to_patches(np.array([[y, x]]), output_size[0], load_size)[0]
    idx = int(y_patch * output_size[0] + x_patch)

    # Diffusion Hyperfeatures
    y2, x2 = points2_dhf[idx]
    dhf_points.append((y2, x2))
    if show_dhf:
      color = "g"
      circ1_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
      circ1_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
      axes[1].add_patch(circ1_1)
      axes[1].add_patch(circ1_2)

  if event.inaxes == axes[1]:
    color = "r"
    circ1_1 = plt.Circle((x, y), radius1, facecolor=color, edgecolor='white', alpha=0.5)
    circ1_2 = plt.Circle((x, y), radius2, facecolor=color, edgecolor='white')
    axes[1].add_patch(circ1_1)
    axes[1].add_patch(circ1_2)
    user_target_points.append((y, x))

  fig.canvas.draw()


fig.canvas.mpl_connect('button_press_event', mouse_event)

def log_predicted_points(title, prefix, predicted_points, user_source_points, user_target_points, load_size, img1, img2, df):
  distances, pck_metric = compute_pck(predicted_points, user_target_points, load_size)
  title = f"{title} \n PCK@0.1: {pck_metric.round(decimals=2)}"
  plt.clf()
  draw_correspondences(user_source_points, predicted_points, img1, img2, "", "", title=title, radius1=8, radius2=1)
  plt.show()
  df.update({
    f"{prefix}_points_y": predicted_points[:, 0],
    f"{prefix}_points_x": predicted_points[:, 1],
    f"{prefix}_distances": distances
  })
  return df

if len(user_target_points) > 0 and len(user_source_points) == len(user_target_points):
  user_source_points, user_target_points = np.array(user_source_points), np.array(user_target_points)
  dhf_points = np.array(dhf_points)
  df = {
    "hash_name": [hash_name] * len(user_source_points),
    "user_source_points_y": user_source_points[:, 0],
    "user_source_points_x": user_source_points[:, 1],
    "user_target_points_y": user_target_points[:, 0],
    "user_target_points_x": user_target_points[:, 1]
  }

  # Log correspondence plots
  plt.clf()
  title = "User Annotations"
  draw_correspondences(user_source_points, user_target_points, img1, img2, "", "", title=title, radius1=8, radius2=1)
  plt.show()
  df = log_predicted_points("Diffusion Hyperfeatures", "dhf", dhf_points, user_source_points, user_target_points, load_size, img1, img2, df)

  if len(user_target_points) > 0 and len(user_source_points) == len(user_target_points):
    # Log PCK distances
    df = pd.DataFrame(df)
    display(df)

    # Log images
    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(synthetic_images[0])
    axs[0].axis("off")
    axs[1].imshow(synthetic_images[1])
    axs[1].axis("off")
    fig.suptitle(prompt)
    fig.tight_layout()
    plt.show()

    # Save the noise used for this run
    if not os.path.exists("seeds"):
      os.mkdir("seeds")
    seeds = outputs[0]
    torch.save(seeds, f"seeds/{hash_name}.pt")