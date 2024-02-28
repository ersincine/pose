import os
import numpy as np
import torch
from kornia.geometry.conversions import matrix4x4_to_Rt, rotation_matrix_to_quaternion
from kornia.geometry.epipolar import relative_camera_motion


def read_matrix4x4(log_path: str, img_no: int) -> torch.Tensor:
    # Important: Image numbers start from 1. But image indices start from 0.
    assert img_no >= 1
    img_idx = img_no - 1

    with open(log_path, "r") as f:
        lines = f.read().splitlines()
    
    line_idx = img_idx * 5
    assert lines[line_idx] == f"{img_idx} {img_idx} 0"
    lines = lines[line_idx + 1: line_idx + 5]
    matrix = torch.tensor([list(map(float, line.split())) for line in lines])
    assert matrix.shape == (4, 4)
    return matrix


def get_relative_camera_motion_from_matrix4x4(matrix1: torch.Tensor, matrix2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert matrix1.shape == (4, 4)
    assert matrix2.shape == (4, 4)

    matrix1 = matrix1.reshape(1, 4, 4)
    matrix2 = matrix2.reshape(1, 4, 4)

    R1, t1 = matrix4x4_to_Rt(matrix1)
    R2, t2 = matrix4x4_to_Rt(matrix2)

    assert R1.shape == (1, 3, 3)
    assert t1.shape == (1, 3, 1)
    assert R2.shape == (1, 3, 3)
    assert t2.shape == (1, 3, 1)

    R, t = relative_camera_motion(R1, t1, R2, t2)
    return R, t


def extract_pose(img1_no: int, img2_no: int, dataset: str, save: bool=True) -> tuple[torch.Tensor, torch.Tensor]:
    assert img1_no >= 1
    assert img2_no >= 1
    assert img1_no != img2_no

    if dataset == "Barn":
        assert img1_no <= 410
    elif dataset == "Truck":
        assert img1_no <= 251
    elif dataset == "Meetingroom":
        assert img1_no <= 371
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    matrix1 = read_matrix4x4(f"datasets/{dataset}_COLMAP_SfM.log", img1_no)
    matrix2 = read_matrix4x4(f"datasets/{dataset}_COLMAP_SfM.log", img2_no)
    R, t = get_relative_camera_motion_from_matrix4x4(matrix1, matrix2)

    q = rotation_matrix_to_quaternion(R)

    t = t.reshape(3)
    q = q.reshape(4)

    if save:
        np.savetxt(f"poses/{dataset}/q-{img1_no}-{img2_no}.txt", q.numpy())
        np.savetxt(f"poses/{dataset}/t-{img1_no}-{img2_no}.txt", t.numpy())
        
    return q, t
