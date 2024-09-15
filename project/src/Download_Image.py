import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve
from utils import download_images

# Create directories to store images
train_image_dir = 'images/train'
# os.makedirs(train_image_dir, exist_ok=True)

# # Function to download an image
# def download_image(url, save_dir):
#     filename = os.path.join(save_dir, os.path.basename(url))
#     try:
#         urlretrieve(url, filename)
#     except Exception as e:
#         print(f"Failed to download {url}: {e}")
#
# # Download images in batches using multithreading
# def download_images(image_links, save_dir, num_workers=8):
#     with ThreadPoolExecutor(max_workers=num_workers) as executor:
#         executor.map(lambda url: download_image(url, save_dir), image_links)

# Load dataset
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/testgit.csv")

# Download train images
train_image_links = train_df['image_link'].tolist()
download_images(train_image_links, train_image_dir , allow_multiprocessing=True)

# Similarly for test images
test_image_dir = 'images/test'
# os.makedirs(test_image_dir, exist_ok=True)
test_image_links = test_df['image_link'].tolist()
download_images(test_image_links, test_image_dir , allow_multiprocessing=True)
