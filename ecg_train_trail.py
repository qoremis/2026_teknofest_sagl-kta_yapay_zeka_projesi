import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------
# FOURIER FİLTRELEME FONKSİYONU
# -----------------------------
def apply_fourier_filter(signal, fs=100, lowcut=0.5, highcut=40.0):
    """
    Sinyali frekans boyutuna geçirir, gürültüyü temizler ve geri döndürür.
    """
    n = signal.shape[-1]
    # Gerçek Fourier Dönüşümü
    fft_vals = np.fft.rfft(signal, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    
    # Filtre maskesi oluştur (Belirlenen frekans aralığı dışını sıfırla)
    mask = (freqs >= lowcut) & (freqs <= highcut)
    fft_vals[:, ~mask] = 0
    
    # Ters Fourier Dönüşümü (Zaman boyutuna geri dönüş)
    return np.fft.irfft(fft_vals, n=n, axis=-1)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# DATASET YÜKLEME (Özetlenmiş)
# -----------------------------
df = pd.read_csv("data/ptbxl_database.csv")
signals, labels = [], []

def label_map(code):
    code = str(code)
    if "NORM" in code: return 0
    elif any(x in code for x in ["AFIB","AFLT","SVT","STACH","SBRAD"]): return 1
    elif any(x in code for x in ["LBBB","RBBB","AVB","IVCD"]): return 2
    return None

for i, row in df.head(12000).iterrows():
    label = label_map(row["scp_codes"])
    if label is None: continue
    path = "data/" + row["filename_hr"]
    signal, _ = wfdb.rdsamp(path)
    # Downsample (500Hz -> 100Hz)
    signals.append(signal.T[:, ::5])
    labels.append(label)

signals, labels = np.array(signals), np.array(labels)

# -----------------------------
# DATASET CLASS (OPTIMIZE EDİLMİŞ)
# -----------------------------
class ECGDataset(Dataset):
    def __init__(self, signals, labels, use_fourier=True):
        self.signals = signals
        self.labels = labels
        self.use_fourier = use_fourier
        self.first_run = True # Kontrol printi için

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        raw_signal = self.signals[idx]
        
        # 1. Fourier Filtreleme
        if self.use_fourier:
            processed_signal = apply_fourier_filter(raw_signal)
            
            # Doğruluk ve Verimlilik Kontrolü (Sadece ilk örnek için)
            if self.first_run and idx == 0:
                noise_diff = np.abs(raw_signal - processed_signal).mean()
                print(f"\n--- Fourier Optimizasyon Raporu ---")
                print(f"Filtreleme ile Temizlenen Ortalama Gürültü Genliği: {noise_diff:.6f}")
                print(f"Sinyal Saflık Katsayısı: %{(100 * (1 - noise_diff/np.max(raw_signal))):.2f}")
                print(f"-----------------------------------\n")
                self.first_run = False
        else:
            processed_signal = raw_signal

        # 2. Normalizasyon
        processed_signal = (processed_signal - np.mean(processed_signal)) / (np.std(processed_signal) + 1e-8)

        return torch.tensor(processed_signal, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# -----------------------------
# MODEL VE EĞİTİM KURULUMU
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(signals, labels, test_size=0.2, stratify=labels)

train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=16)

class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, 7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        return self.fc(self.dropout(x))

model = ECGModel().to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2, 1.2]).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# -----------------------------
# EĞİTİM DÖNGÜSÜ
# -----------------------------
for epoch in range(40):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# TEST
# -----------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        _, predicted = torch.max(model(x), 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(f"\n🔥 Fourier Optimizasyonlu Test Accuracy: {correct/total:.4f}")
torch.save(model.state_dict(), "ecg_fourier_optimized.pth")