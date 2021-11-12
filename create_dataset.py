import os
import glob
import shutil
import numpy as np

base_dir = "/home/guowei/Research/View-Synthesis-Current-Works/Neural-Scene-Flow-Fields/nerf_data/"
scene_name = "scene003"

output_name= "iccv-03"

output_dir = os.path.join(base_dir, output_name, "dense")
output_image_dir = os.path.join(base_dir, output_name, "dense", "images")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)

original_poses = np.load(os.path.join(base_dir, scene_name, "poses_bounds.npy"))

# copy and organize image files 
cams = [4,5]
num_cams = len(cams)
frame_range = [0, 120]

new_poses = []
for i, frame_idx in enumerate(range(frame_range[0], frame_range[1])):
	cam_idx = cams[i % 2]
	cam_img_path = os.path.join(base_dir, scene_name, "cam0{}".format(cam_idx), "*.jpg")
	images = sorted(glob.glob(cam_img_path))[frame_range[0]:frame_range[1]]
	shutil.copy2(images[i], os.path.join(output_image_dir, "{:05d}.jpg".format(i)))
	new_poses.append(original_poses[cam_idx])

new_poses = np.array(new_poses)
np.save(os.path.join(base_dir, output_dir, "poses_bounds.npy"), new_poses)
