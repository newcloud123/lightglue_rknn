import torch
import torch.nn.functional as F

from .unet import Unet


def heatmap_to_keypoints(heatmap: torch.Tensor, n: int, window_size: int = 5):
    # NMS
    b, _, h, w = heatmap.shape
    mask = F.max_pool2d(heatmap, kernel_size=window_size, stride=1, padding=window_size // 2)
    heatmap = torch.where(heatmap == mask, heatmap, torch.zeros_like(heatmap))

    # Select top-K
    top_scores, top_indices = heatmap.reshape(b, h * w).topk(n)
    if torch.jit.is_tracing():  # type: ignore
        one = torch.tensor(1)  # Always constant, safe to ignore warning.
        top_indices = top_indices.unsqueeze(2).floor_divide(
            torch.stack([w, one]).to(device=top_indices.device)  # type: ignore
        ) % torch.stack([h, w]).to(device=top_indices.device)  # type: ignore
    else:
        top_indices = top_indices.unsqueeze(2).floor_divide(
            torch.tensor([w, 1], device=top_indices.device)
        ) % torch.tensor([h, w], device=top_indices.device)
    top_keypoints = top_indices.flip(2)

    return top_keypoints, top_scores


class DISK(torch.nn.Module):
    url = "https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth"

    def __init__(self, descriptor_dim: int = 128, nms_window_size: int = 5, num_keypoints: int = 1024) -> None:
        super().__init__()
        if nms_window_size % 2 != 1:
            raise ValueError(f"window_size has to be odd, got {nms_window_size}")

        self.descriptor_dim = descriptor_dim
        self.nms_window_size = nms_window_size
        self.num_keypoints = num_keypoints

        self.unet = Unet(in_features=3, size=5, down=[16, 32, 64, 64, 64], up=[64, 64, 64, descriptor_dim + 1])

        self.load_state_dict(torch.hub.load_state_dict_from_url(self.url)["extractor"])

    def forward(
        self,
        image: torch.Tensor,  # (B, 3, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = image.shape[0]

        unet_output: torch.Tensor = self.unet(image)
        descriptors = unet_output[:, : self.descriptor_dim]  # (B, D, H, W)
        heatmaps = unet_output[:, self.descriptor_dim :]  # (B, 1, H, W)

        keypoints, scores = heatmap_to_keypoints(heatmaps, n=self.num_keypoints, window_size=self.nms_window_size)

        descriptors = descriptors.permute(0, 2, 3, 1)
        batches = torch.arange(b, device=image.device)[:, None].expand(b, self.num_keypoints)
        descriptors = descriptors[(batches, keypoints[:, :, 1], keypoints[:, :, 0])]
        descriptors = F.normalize(descriptors, dim=-1)

        return (
            keypoints,  # (B, N, 2) with <X, Y>
            scores,  # (B, N)
            descriptors,  # (B, N, descriptor_dim)
        )
