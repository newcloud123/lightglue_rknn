import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..ops import multi_head_attention_dispatch

torch.backends.cudnn.deterministic = True


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, descriptor_dim: int, num_heads: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = descriptor_dim // num_heads
        self.Wr = nn.Linear(M, head_dim // 2, bias=False)
        self.gamma = gamma
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines])
        return emb.repeat_interleave(2, dim=3).repeat(1, 1, 1, self.num_heads).unsqueeze(4)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )


class SelfBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv: torch.Tensor = self.Wqkv(x)
        qkv = qkv.reshape((b, n, self.embed_dim, 3))
        qk, v = qkv[..., :2], qkv[..., 2]
        qk = self.apply_cached_rotary_emb(encoding, qk)
        q, k = qk[..., 0], qk[..., 1]
        context = multi_head_attention_dispatch(q, k, v, self.num_heads)
        message = self.out_proj(context)
        return x + self.ffn(torch.concat([x, message], 2))

    def rotate_half(self, qk: torch.Tensor) -> torch.Tensor:
        b, n, _, _ = qk.shape
        qk = qk.reshape((b, n, self.num_heads, self.head_dim // 2, 2, 2))
        qk = torch.stack((-qk[..., 1, :], qk[..., 0, :]), dim=4)
        qk = qk.reshape((b, n, self.embed_dim, 2))
        return qk

    def apply_cached_rotary_emb(self, encoding: torch.Tensor, qk: torch.Tensor) -> torch.Tensor:
        return qk * encoding[0] + self.rotate_half(qk) * encoding[1]


class CrossBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        b, _, _ = descriptors.shape
        qk, v = self.to_qk(descriptors), self.to_v(descriptors)

        indices = torch.arange(b, device=descriptors.device)
        swap = (indices // 2) * 2 + (1 - indices % 2)  # swap trick
        m = multi_head_attention_dispatch(qk, qk[swap], v[swap], self.num_heads)
        m = self.to_out(m)
        descriptors = descriptors + self.ffn(torch.concat([descriptors, m], 2))
        return descriptors


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads)
        self.cross_attn = CrossBlock(embed_dim, num_heads)

    def forward(self, descriptors: torch.Tensor, encodings: torch.Tensor) -> torch.Tensor:
        descriptors = self.self_attn(descriptors, encodings)
        return self.cross_attn(descriptors)


def sigmoid_log_double_softmax(similarities: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    certainties = F.logsigmoid(z[0::2]) + F.logsigmoid(z[1::2]).transpose(1, 2)
    scores0 = F.log_softmax(similarities, 2)
    scores1 = F.log_softmax(similarities, 1)
    scores = scores0 + scores1 + certainties
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.25
        self.final_proj = nn.Linear(dim, dim, bias=True)
        self.matchability = nn.Linear(dim, 1, bias=True)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """build assignment matrix from descriptors"""
        mdescriptors = self.final_proj(descriptors) / self.scale
        similarities = mdescriptors[0::2] @ mdescriptors[1::2].transpose(1, 2)
        z = self.matchability(descriptors)
        scores = sigmoid_log_double_softmax(similarities, z)
        return scores

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)

def filter_matches1_rknn(
    scores: torch.Tensor,
    threshold: float,
    max_matches: int = 50,   # RK3588 推荐
):
    """
    RKNN-friendly LightGlue match filtering.
    Fixed-shape, no NonZero / where.
    """

    B, N0, N1 = scores.shape  # [1,1024,1024]

    # 1. mutual nearest neighbor
    max0 = torch.topk(scores, k=1, dim=2, sorted=False)
    max1 = torch.topk(scores, k=1, dim=1, sorted=False)

    m0 = max0.indices[:, :, 0]      # [B, 1024]
    m1 = max1.indices[:, 0, :]      # [B, 1024]

    indices = torch.arange(N0, device=scores.device)[None, :].expand(B, N0)
    mutual = indices == m1.gather(1, m0)

    # 2. score & threshold
    mscores = max0.values[:, :, 0].exp()
    valid = (mscores > threshold) & mutual

    # 3. mask instead of where
    masked_scores = mscores * valid.float()

    # 4. fixed TopK (critical for RKNN)
    topk_scores, topk_idx = torch.topk(
        masked_scores,
        k=max_matches,
        dim=1,
        sorted=False
    )

    # 5. corresponding indices
    topk_m1 = m0.gather(1, topk_idx)

    # 6. batch index
    b_idx = torch.arange(B, device=scores.device)[:, None].expand(B, max_matches)

    # 7. fixed-shape outputs
    matches = torch.stack([b_idx, topk_idx, topk_m1], dim=2)
    mscores = topk_scores

    return matches[0], mscores[0]

def filter_matches1(scores: torch.Tensor, threshold: float):
    max0 = torch.topk(scores, k=1, dim=2, sorted=False)  # scores.max(2) 
    max1 = torch.topk(scores, k=1, dim=1, sorted=False)  # scores.max(1)
    m0, m1 = max0.indices[:, :, 0], max1.indices[:, 0, :]
    print("m0 shape = {} | m1 shape = {} ".format(m0.shape,m1.shape))  # [1,1024]  [1,1024]
    indices = torch.arange(m0.shape[1], device=m0.device).expand_as(m0)
    print("indices shape = ",indices.shape)  # [1,1024]
    mutual = indices == m1.gather(1, m0)
    print("mutual shape = ",mutual.shape)    # [1,1024]
    print("max0.values shape = {} | max1.values = {} ".format(max0.values.shape,max1.values.shape))  # ([1, 1024, 1]) | [1, 1, 1024]) 
    mscores = max0.values[:, :, 0].exp()

    valid = mscores > threshold

    b_idx, m0_idx = torch.where(valid & mutual)
    m1_idx = m0[b_idx, m0_idx]
    matches = torch.concat([b_idx[:, None], m0_idx[:, None], m1_idx[:, None]], 1)
    mscores = mscores[b_idx, m0_idx]
    return matches, mscores


def show_mismatch_keys(model, state_dict, device='cpu'):
    """
    打印 model 与 ckpt 之间匹配不上的权重 key
    :param model: 你的网络实例
    :param ckpt_path: 保存的 .pth/.pt 文件路径
    :param device: 加载位置
    """
    # 1. 加载 checkpoint
    state_dict = state_dict # torch.load(ckpt_path, map_location=device)
    # 如果 checkpoint 是嵌套字典，按需取 key，例如 state_dict['model']
    if 'model' in state_dict:
        state_dict = state_dict['model']

    # 2. 模拟 load_state_dict 的严格检查过程
    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(state_dict.keys())

    missing_keys   = model_keys - ckpt_keys          # 模型有，ckpt 没有
    unexpected_keys = ckpt_keys - model_keys         # ckpt 有，模型没有

    print('-----------  Missing in checkpoint  -----------')
    for k in sorted(missing_keys):
        print(k)

    print('\n-----------  Unexpected in checkpoint  -----------')
    for k in sorted(unexpected_keys):
        print(k)

    # 3. 如果形状不匹配，也可以进一步检查
    shape_mismatch = []
    for k in model_keys & ckpt_keys:
        if model.state_dict()[k].shape != state_dict[k].shape:
            shape_mismatch.append(
                (k, model.state_dict()[k].shape, state_dict[k].shape)
            )
    if shape_mismatch:
        print('\n-----------  Shape mismatch  -----------')
        for k, m_shape, c_shape in shape_mismatch:
            print(f'{k}: model {m_shape} vs ckpt {c_shape}')
            
class LightGlue(nn.Module):
    def __init__(
        self,
        url: str,
        input_dim: int = 256,
        descriptor_dim: int = 256,
        num_heads: int = 4,
        n_layers: int = 9,
        filter_threshold: float = 0.1,  # match threshold
        depth_confidence: float = -1,  # -1 is no early stopping, recommend: 0.95
        width_confidence: float = -1,  # -1 is no point pruning, recommend: 0.99
    ) -> None:
        super().__init__()

        self.descriptor_dim = descriptor_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.filter_threshold = filter_threshold
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence

        if input_dim != self.descriptor_dim:
            self.input_proj = nn.Linear(input_dim, self.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        self.posenc = LearnableFourierPositionalEncoding(2, self.descriptor_dim, self.num_heads)

        d, h, n = self.descriptor_dim, self.num_heads, self.n_layers

        self.transformers = nn.ModuleList([TransformerLayer(d, h) for _ in range(n)])

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])

        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])
        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor([self.confidence_threshold(i) for i in range(n)]),
        )
        """
        state_dict = torch.hub.load_state_dict_from_url(url)
        # rename old state dict entries
        for i in range(n):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        """
        state_dict = torch.load("/data/luoshiyong/code/LightGlue-ONNX-main/gim_lightglue_100h.ckpt", map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('superpoint.'):
                state_dict.pop(k)
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        show_mismatch_keys(self,state_dict)
        self.load_state_dict(state_dict, strict=False)
      # 新增：重写的 filter_matches 类方法
    def filter_matches(self, scores: torch.Tensor, threshold: float): # [1, 1024, 1024]  0.1
        B, N, _ = scores.shape  # 1, 1024 
        max0, max1 = scores.max(2), scores.max(1)  # 
        print("max0 shape = ",max0.values.shape)  # [1,1024]
        print("max1 shape = ",max1.values.shape)  # [1,1024]
        m0, m1 = max0.indices, max1.indices
        
        arange = torch.arange(N, device=scores.device).view(1, -1).repeat(B, 1)
        mutual = m1.gather(1, m0) == arange
        mutual_expand = mutual.unsqueeze(-1).repeat(1, 1, N)
        print("mutual_expand = ",mutual_expand.shape)  # [1,1024,1024]

        threshold_mask = scores.exp() > threshold
        final_mask = mutual_expand & threshold_mask
        filtered_scores = scores * final_mask.float()
        return final_mask, filtered_scores
    def forward(
        self,
        keypoints: torch.Tensor,  # (2B, N, 2), normalized
        descriptors: torch.Tensor,  # (2B, N, D)
    ):
        descriptors = self.input_proj(descriptors)

        # positional embeddings
        encodings = self.posenc(keypoints)  # (2, 2B, *, 64, 1)

        # GNN + final_proj + assignment
        for i in range(self.n_layers):
            # self+cross attention
            descriptors = self.transformers[i](descriptors, encodings)

        scores = self.log_assignment[i](descriptors)  # (B, N, N)
        print("scores shape =  ",scores.shape)      # [1, 1024, 1024]
        print("self.filter_threshold = ",self.filter_threshold) #0.1
        print("mscores max = ", scores.max())
        # matches, mscores = self.filter_matches(scores, self.filter_threshold)
        # matches, mscores = filter_matches1(scores, self.filter_threshold)
        matches,mscores = filter_matches1_rknn(scores, self.filter_threshold)
        print("matches shape = ",matches.shape)
        print("mscores shape = ",mscores.shape)

        
        return matches, mscores  # (M, 3), (M,)

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self,
        confidences: torch.Tensor | None,
        scores: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.depth_confidence
