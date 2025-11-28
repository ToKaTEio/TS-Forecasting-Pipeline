import torch
import torch.nn as nn

class MovingAverage(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)
    
    def forward(self, x):
        # x[batch_size, seq_len, features]
        P = (self.avg.kernel_size[0] - 1) // 2
        
        front_pad = x[:, :P, :].flip(1)
        rear_pad = x[:, -P:, :].flip(1)
        x_pad = torch.cat([front_pad, x, rear_pad], dim=1)
        
        return self.avg(x_pad.permute(0,2,1)).permute(0,2,1)


class TrendBlock(nn.Module):
    def __init__(self, input_len, pred_len):
        super().__init__()
        self.linear1 = nn.Linear(input_len, input_len//2)
        self.linear2 = nn.Linear(input_len//2, pred_len)
    
    def forward(self, x):
        return self.linear2(self.linear1(x)) + x.mean(dim=1, keepdim=True)  # residual


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)
    
    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    def __init__(self, input_len=336, pred_len=24, num_features=7):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=25)
        
        self.linear_seasonal = nn.ModuleList([
            nn.Linear(input_len, pred_len) for _ in range(num_features)
        ])
        self.trend_blocks = nn.ModuleList([
            TrendBlock(input_len, pred_len) for _ in range(num_features)
        ])
        
    def forward(self, x):
        # x: [batch, len=336, features=7]
        seasonal, trend = self.decomp(x)
        
        pred_seasonal = torch.stack([
            layer(seasonal[:, :, i]) for i, layer in enumerate(self.linear_seasonal)
        ], dim=2)
        pred_trend = torch.stack([
            layer(trend[:, :, i]) for i, layer in enumerate(self.trend_blocks)
        ], dim=2)
        
        return pred_seasonal[:, :, -1] + pred_trend[:, :, -1]  # [batch, len=24, features=7]