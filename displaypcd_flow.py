
import open3d as o3d
import os
import numpy as np
import imageio
from PIL import Image
import imageio.v2 as imageio
import re


def to_o3d_pcd(np_array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_array)
    return pcd


def capture_and_save_images_from_pcd(flow_file, image_directory):
        os.makedirs(image_directory, exist_ok=True)
        flow = np.load(flow_file)
        for i in range(len(flow)):
            
            pcd = to_o3d_pcd(flow[i])
            # Visualize the point cloud and capture the image
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)  # Set to True if you want to see the window
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Capture and save the image
            image_path = os.path.join(image_directory, '{}.png'.format(i))
            vis.capture_screen_image(image_path, do_render=True)
            vis.destroy_window()

# Assuming filenames are like 'image1.png', 'image2.png', ..., 'image1000.png'
# Extract the numerical part of each filename for sorting
def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

# image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png')], key=extract_number)

def make_video(image_directory, video_path, fps=10):
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png')], key=extract_number)
    writer = imageio.get_writer(video_path, fps=fps)

    for image_file in image_files:
        # print(image_file)
        image_path = os.path.join(image_directory, image_file)
        image = imageio.imread(image_path)
        writer.append_data(image)
    writer.close()
    print(f"Video saved at {video_path}")




def make_gif(image_directory, gif_path, fps=10):
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.png')], key=extract_number)
    images = [imageio.imread(os.path.join(image_directory, f)) for f in image_files]
    imageio.mimsave(gif_path, images, fps=fps)
    print(f"GIF saved at {gif_path}")


# Define your directories
flow_file = './traj/traj.npy'  # Directory containing PLY files
image_directory = './traj/images'  # Temporary directory for images
gif_path = './traj/trajectory_animation.gif'  # Output GIF path

# Capture and save images from each PLY file
capture_and_save_images_from_pcd(flow_file, image_directory)
video_path = './traj/trajectory_animation.mp4'  # Output video path

# Create a video from the saved images
make_video(image_directory, video_path, fps=50)
# Create a GIF from the saved images
# make_gif(image_directory, gif_path, fps=100)
