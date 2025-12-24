import torch 
class Pipeline2(torch.nn.Module):
    def __init__(self, extractor: torch.nn.Module):
        super().__init__()
        self.extractor = extractor

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _, _, h, w = images.shape
        # Extract keypoints and features
        keypoints, scores, descriptors = self.extractor(images)
        # Normalize keypoints
        if torch.jit.is_tracing():  # type: ignore
            size = torch.stack([w, h]).to(device=keypoints.device)  # type: ignore
        else:
            size = torch.tensor([w, h], device=images.device)
        normalized_keypoints = 2 * keypoints / size - 1
        # Match keypoints
        return keypoints,descriptors
        
        # matches, mscores = self.matcher(normalized_keypoints, descriptors)
        # return keypoints, matches, mscores