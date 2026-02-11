import torch
import kornia

def decode_heatmap(heatmaps):
    """
    히트맵 -> 좌표
    """
    # Soft Argmax
    # temperature가 높을수록 argmax에 가까워지고, 낮을수록 평균에 가까워짐
    coords = kornia.geometry.subpix.spatial_soft_argmax2d(
        input=heatmaps,
        temperature=torch.tensor(1.0, device=heatmaps.device),
        normalized_coordinates=False
    )
    return coords