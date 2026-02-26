# src/rlhf/pointwise_reward_model.py
"""
pointwise_reward_model.py
Input: concat(q, p, q-p, q*p) -> emb_dim*4 -> MLP -> score escalar
"""
import torch
import torch.nn as nn


class PointwiseRewardModel(nn.Module):
    def __init__(self, emb_dim: int = 384, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.emb_dim    = emb_dim
        self.hidden_dim = hidden_dim
        in_dim = emb_dim * 4  # [q; p; q-p; q*p]

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self._init_weights()
        import logging
        logging.getLogger(__name__).info(
            f"PointwiseRewardModel: emb={emb_dim}, hidden={hidden_dim}, "
            f"input={in_dim}, params={sum(p.numel() for p in self.parameters()):,}"
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if q.dim() == 1: q = q.unsqueeze(0)
        if p.dim() == 1: p = p.unsqueeze(0)
        if q.size(0) == 1 and p.size(0) > 1:
            q = q.expand(p.size(0), -1)
        x = torch.cat([q, p, q - p, q * p], dim=-1)
        return self.mlp(x).squeeze(-1)

    def score_single(self, q: torch.Tensor, p: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            s = self(q, p)
        return s.item() if s.dim() == 0 else s[0].item()

    def save(self, path: str):
        torch.save({'model_state': self.state_dict(),
                    'config': {'emb_dim': self.emb_dim,
                               'hidden_dim': self.hidden_dim,
                               'model_type': 'pointwise'}}, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PointwiseRewardModel':
        ckpt = torch.load(path, map_location=device)
        c    = ckpt.get('config', {})
        m    = cls(emb_dim=c.get('emb_dim', 384), hidden_dim=c.get('hidden_dim', 256))
        m.load_state_dict(ckpt['model_state'])
        return m.to(device)


class PointwiseMarginLoss(nn.Module):
    def __init__(self, base_margin: float = 1.0):
        super().__init__()
        self.base_margin = base_margin

    def forward(self, s_a, s_b, diffs, weights=None):
        margins = (diffs / 3.0) * self.base_margin
        losses  = torch.clamp(margins - s_a + s_b, min=0)
        if weights is not None:
            losses = losses * weights
        return losses.mean()