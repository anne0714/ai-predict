import torch.nn as nn
import torch.nn.functional as F

class CNNTransformer(nn.Module):
    def __init__(self):
        super(CNNTransformer, self).__init__()

        # CNN 部分
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # → [32, 28, 28]
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # → [64, 28, 28]
        self.pool = nn.MaxPool2d(2, 2)  # → [64, 14, 14]

        # 將 CNN 輸出 reshape 為 Transformer 輸入：[batch, seq_len, d_model]
        self.linear_in = nn.Linear(14 * 14, 128)  # 每個 channel 當作一個 sequence item
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        # 最後接分類器
        self.fc = nn.Linear(64 * 128, 10)  # 64 是 token 數量（channel 數），128 是每 token 的向量長度

    def forward(self, x):
        x = F.relu(self.conv1(x))     # [batch, 32, 28, 28]
        x = F.relu(self.conv2(x))     # [batch, 64, 28, 28]
        x = self.pool(x)              # [batch, 64, 14, 14]

        # 將每個 channel 當成一個 token，flatten spatial 維度
        x = x.view(x.size(0), 64, -1)           # [batch, 64, 14*14]
        x = self.linear_in(x)                   # [batch, 64, 128]
        x = x.permute(1, 0, 2)                  # → [seq_len=64, batch, d_model=128]

        x = self.transformer(x)                 # → [64, batch, 128]
        x = x.permute(1, 0, 2)                  # → [batch, 64, 128]
        x = x.reshape(x.size(0), -1)            # → [batch, 64*128]

        x = self.fc(x)                          # → [batch, 10]
        return x