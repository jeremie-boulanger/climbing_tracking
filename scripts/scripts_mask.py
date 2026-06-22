import numpy as np
import cv2
import os
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import torch

def get_homography(image_destination, image_source):
    """
    Calculate the homography matrix between two images using SuperPoint and LightGlue.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    image0 = load_image(image_source).to(device)
    image1 = load_image(image_destination).to(device)

    with torch.no_grad():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        matches01 = matcher({'image0': feats0, 'image1': feats1})

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]] 
    matches = matches01['matches']
    points_src = feats0['keypoints'][matches[..., 0]].cpu().numpy()
    points_dst = feats1['keypoints'][matches[..., 1]].cpu().numpy()
    h, mask = cv2.findHomography(points_src, points_dst, cv2.RANSAC, 5.0)
    
    return h

def transfo_mask_folder(mask_folder, image_destination, image_source, output_folder):
    """Apply transform to all masks"""

    h = get_homography(image_destination, image_source)
 
    img_dst = cv2.imread(image_destination)
    if img_dst is None:
        raise FileNotFoundError(f"Destination image not found: {image_destination}")
 
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith(".png")]
 
    if not mask_files:
        raise ValueError(f"No PNG files found in: {mask_folder}")
 
    os.makedirs(output_folder, exist_ok=True)
 
    processed = []
    for mask in mask_files:
        pts_src = os.path.join(mask_folder, mask)
        img_src = cv2.imread(pts_src)
        if img_src is None:
            print(f"Cannot read: {pts_src}")
            continue
        output = cv2.warpPerspective(img_src, h, (img_dst.shape[1], img_dst.shape[0]))
        out_path = os.path.join(output_folder, mask)
        cv2.imwrite(out_path, output)
        processed.append(out_path)
 
    return processed