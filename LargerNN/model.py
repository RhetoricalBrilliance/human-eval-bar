
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Residual block tailored for tabular MLPs ---
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_mult=2, dropout=0.15):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()  # smooth ReLU; robust on tabular
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.act(out)
        return residual + out  # skip connection

class TabResNetModel(nn.Module):
    """
    A deeper ResNet-style MLP for tabular classification.
    Strong baseline per Gorishniy et al. (NeurIPS 2021).
    """
    def __init__(self, input_dim: int, num_classes=3,
                 stem_width=256, num_blocks=6,
                 block_mult=1.5, dropout=0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, stem_width),
            nn.BatchNorm1d(stem_width),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualMLPBlock(stem_width, hidden_mult=block_mult, dropout=dropout))
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(stem_width, stem_width // 2),
            nn.BatchNorm1d(stem_width // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(stem_width // 2, num_classes)
        )

        # Kaiming init is a solid default for SiLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        return self.head(x)  # logits
