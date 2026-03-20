import wfdb
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# MODEL
# -----------------------------
class ECGModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(12,32,7,padding=3)
        self.conv2 = nn.Conv1d(32,64,5,padding=2)
        self.conv3 = nn.Conv1d(64,128,3,padding=1)

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128,3)

    def forward(self,x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # 🔥 EXPLAINABLE AI için feature map sakla
        self.feature_map = self.pool(self.relu(self.conv3(x)))

        x = self.feature_map

        x = self.global_pool(x)
        x = x.squeeze(-1)

        x = self.dropout(x)
        x = self.fc(x)

        return x

# -----------------------------
# MODEL YÜKLE
# -----------------------------
model = ECGModel().to(device)
model.load_state_dict(torch.load("ecg_final_safe.pth", map_location=device))
model.eval()

# -----------------------------
# SINIFLAR
# -----------------------------
classes = [
    "Normal ECG",
    "Arrhythmia",
    "Conduction Block"
]

# -----------------------------
# DATASET
# -----------------------------
df = pd.read_csv("data/ptbxl_database.csv")

# -----------------------------
# ECG SEÇ
# -----------------------------
while True:

    row = df.sample(1).iloc[0]
    path = "data/" + row["filename_hr"]

    try:
        signal, _ = wfdb.rdsamp(path)

        if signal.shape[1] == 12:
            signal = signal.T

        if signal.shape[0] == 12:
            break

    except:
        continue

print("Dosya:", path)

# -----------------------------
# PREPROCESS
# -----------------------------
signal = signal[:, ::5]
signal = (signal - np.mean(signal)) / (np.std(signal)+1e-8)

x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)

# -----------------------------
# GRADIENT HOOK (EXPLAINABLE AI)
# -----------------------------
gradients = []

def save_grad(grad):
    gradients.append(grad)

# forward
output = model(x)

# hook ekle
model.feature_map.register_hook(save_grad)

# target class
predicted_class = torch.argmax(output)

# backward
model.zero_grad()
output[0, predicted_class].backward()

# -----------------------------
# GRAD-CAM
# -----------------------------
grads = gradients[0].cpu().numpy()[0]          # (C, L)
feature_map = model.feature_map.detach().cpu().numpy()[0]

weights = np.mean(grads, axis=1)               # (C)

cam = np.zeros(feature_map.shape[1])

for i, w in enumerate(weights):
    cam += w * feature_map[i]

cam = np.maximum(cam, 0)
cam = cam / np.max(cam)

# -----------------------------
# TAHMİN
# -----------------------------
probs = torch.softmax(output, dim=1)
predicted = predicted_class.item()

print("\n📊 SONUÇ")
print("Tahmin:", classes[predicted])
print("Olasılıklar:", probs.detach().cpu().numpy())

# -----------------------------
# 📈 ECG GRAFİK
# -----------------------------
plt.figure()

plt.plot(signal[0], label="ECG Lead 1")

# attention overlay
cam_resized = np.interp(
    np.linspace(0, len(cam), len(signal[0])),
    np.arange(len(cam)),
    cam
)

plt.fill_between(
    range(len(signal[0])),
    signal[0],
    where=cam_resized > 0.5,
    alpha=0.3
)

plt.title(f"Tahmin: {classes[predicted]}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

plt.show()