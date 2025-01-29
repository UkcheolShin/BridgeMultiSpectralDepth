from __future__ import division
import torch
import torch.nn.functional as F
from .submodules import check_sizes, pixel2cam, cam2pixel, pose_vec2mat, euler2mat

def inverse_warp(img, depth, pose_mat, intrinsics, intrinsics_inv, \
                 rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, 'img', 'BCHW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose_mat, 'pose_mat', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat # intrinsics.bmm(pose_mat) # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(
        cam_coords, rot, tr, padding_mode)[0]  # [B,H,W,2]
    projected_img = F.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points

def inverse_warp_depth(ref_depth, depth, pose_mat, intrinsics, intrinsics_inv, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose_mat: 6DoF pose parameters from target to source -- [B, 3, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose_mat, 'pose_mat', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel(
        cam_coords, rot, tr, padding_mode)  # [B,H,W,2]

    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_depth, computed_depth, valid_points
    