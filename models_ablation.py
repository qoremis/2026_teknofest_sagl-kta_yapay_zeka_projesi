import torch
import torch.nn as nn

# ==========================================================
# 0. ORTAK KATMANLAR (Tüm modeller kullanır)
# ==========================================================
class SinyalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 4, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 4, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x).permute(0, 2, 1)
        k = self.key(x)
        v = self.value(x)
        scores = torch.bmm(q, k) / ( (x.shape[1] // 4) ** 0.5 )
        return x + torch.bmm(v, self.softmax(scores).permute(0, 2, 1))

# ==========================================================
# 1. RESNET TABANLI HİBRİT MODEL (Gradyan Koruyucu)
# ==========================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ResNetECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(nn.Conv1d(12, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU())
        self.layer1 = ResidualBlock1D(32, 64, stride=2)   # Çıkış: 64
        self.layer2 = ResidualBlock1D(64, 128, stride=2)  # Çıkış: 128
        
        self.attention = SinyalAttention(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.psd_bn = nn.BatchNorm1d(12)
        self.hybrid_norm = nn.LayerNorm(140)
        self.classifier = nn.Sequential(nn.Linear(140, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 5))

    def forward(self, x, psd):
        x = self.layer2(self.layer1(self.init_conv(x)))
        x = torch.squeeze(self.global_pool(self.attention(x)), dim=-1)
        hybrid = self.hybrid_norm(torch.cat((x, self.psd_bn(psd)), dim=1))
        return self.classifier(hybrid)

# ==========================================================
# 2. INCEPTION TABANLI HİBRİT MODEL (Çoklu Ölçek Avcısı)
# ==========================================================
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch):
        super().__init__()
        # Farklı zaman pencereleri için 4 paralel kol (1x, 3x, 5x, 7x)
        self.b1 = nn.Sequential(nn.Conv1d(in_channels, out_channels_per_branch, 1), nn.BatchNorm1d(out_channels_per_branch), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv1d(in_channels, out_channels_per_branch, 3, padding=1), nn.BatchNorm1d(out_channels_per_branch), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv1d(in_channels, out_channels_per_branch, 5, padding=2), nn.BatchNorm1d(out_channels_per_branch), nn.ReLU())
        self.b4 = nn.Sequential(nn.Conv1d(in_channels, out_channels_per_branch, 7, padding=3), nn.BatchNorm1d(out_channels_per_branch), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class InceptionECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(nn.Conv1d(12, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU())
        self.inc1 = InceptionBlock1D(in_channels=32, out_channels_per_branch=16) # 16*4 = 64 kanal çıkış
        self.pool = nn.MaxPool1d(2)
        self.inc2 = InceptionBlock1D(in_channels=64, out_channels_per_branch=32) # 32*4 = 128 kanal çıkış
        
        self.attention = SinyalAttention(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.psd_bn = nn.BatchNorm1d(12)
        self.hybrid_norm = nn.LayerNorm(140)
        self.classifier = nn.Sequential(nn.Linear(140, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 5))

    def forward(self, x, psd):
        x = self.inc2(self.pool(self.inc1(self.init_conv(x))))
        x = torch.squeeze(self.global_pool(self.attention(x)), dim=-1)
        hybrid = self.hybrid_norm(torch.cat((x, self.psd_bn(psd)), dim=1))
        return self.classifier(hybrid)

# ==========================================================
# 3. DENSENET TABANLI HİBRİT MODEL (Öznitelik Birleştirici)
# ==========================================================
class DenseLayer1D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1) # Önceki tüm özellikleri yeniye ekler

class DenseNetECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(nn.Conv1d(12, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU())
        # 32 (Giriş) + 32 (Büyüme) = 64
        self.dense1 = DenseLayer1D(32, 32)
        self.pool = nn.MaxPool1d(2)
        # 64 (Giriş) + 64 (Büyüme) = 128
        self.dense2 = DenseLayer1D(64, 64) 
        
        self.attention = SinyalAttention(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.psd_bn = nn.BatchNorm1d(12)
        self.hybrid_norm = nn.LayerNorm(140)
        self.classifier = nn.Sequential(nn.Linear(140, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 5))

    def forward(self, x, psd):
        x = self.dense2(self.pool(self.dense1(self.init_conv(x))))
        x = torch.squeeze(self.global_pool(self.attention(x)), dim=-1)
        hybrid = self.hybrid_norm(torch.cat((x, self.psd_bn(psd)), dim=1))
        return self.classifier(hybrid)