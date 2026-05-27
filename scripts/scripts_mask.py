import numpy as np
import cv2
import os

def transfo(pts_src, pts_dst) :
    """Calculate Homography"""

    return cv2.getPerspectiveTransform(pts_src, pts_dst)

def transfo_mask(h, src, dst, image_file) : 
    """Warp source image to destination based on homography"""

    img_src = cv2.imread(src)
    img_dst = cv2.imread(dst)
    output = cv2.warpPerspective(img_src, h, (img_dst.shape[1],img_dst.shape[0]))
    os.makedirs(os.path.dirname(image_file), exist_ok=True)
    cv2.imwrite(image_file, output) 
    return 

def transfo_mask_folder(pts_src, pts_dst, mask_folder, image_destination, output_folder):
    """Apply transform to all masks"""

    h = transfo(pts_src, pts_dst)
 
    img_dst = cv2.imread(image_destination)
    if img_dst is None:
        raise FileNotFoundError(f"Destination image not found: {image_destination}")
 
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith(".png")]
 
    if not mask_files:
        raise ValueError(f"No PNG files found in: {mask_folder}")
 
    os.makedirs(output_folder, exist_ok=True)
 
    processed = []
    for mask in mask_files:
        src_path = os.path.join(mask_folder, mask)
        img_src = cv2.imread(src_path)
        if img_src is None:
            print(f"Cannot read: {src_path}")
            continue
        output = cv2.warpPerspective(img_src, h, (img_dst.shape[1], img_dst.shape[0]))
        out_path = os.path.join(output_folder, mask)
        cv2.imwrite(out_path, output)
        processed.append(out_path)
 
    return processed