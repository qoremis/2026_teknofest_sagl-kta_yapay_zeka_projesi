import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# DATASET
# -----------------------------
df = pd.read_csv("data/ptbxl_database.csv")

signals = []
labels = []

def label_map(code):

    code = str(code)

    # NORMAL
    if "NORM" in code:
        return 0

    # ARRHYTHMIA
    elif any(x in code for x in ["AFIB","AFLT","SVT","STACH","SBRAD"]):
        return 1

    # CONDUCTION BLOCK
    elif any(x in code for x in ["LBBB","RBBB","AVB","IVCD"]):
        return 2

    else:
        return None


print("Loading ECG...")

for i,row in df.head(12000).iterrows():

    label = label_map(row["scp_codes"])

    if label is None:
        continue

    path = "data/" + row["filename_hr"]

    signal,_ = wfdb.rdsamp(path)

    signal = signal.T

    # 🔥 Downsample (5000 → 1000)
    signal = signal[:, ::5]

    signals.append(signal)
    labels.append(label)


signals = np.array(signals)
labels = np.array(labels)

print("Dataset:", signals.shape)

# -----------------------------
# SPLIT
# -----------------------------
X_train,X_test,y_train,y_test = train_test_split(
    signals,labels,test_size=0.2,stratify=labels
)

# -----------------------------
# DATASET CLASS
# -----------------------------
class ECGDataset(Dataset):

    def __init__(self,signals,labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self,idx):

        signal = self.signals[idx]

        # normalize
        signal = (signal - np.mean(signal)) / (np.std(signal)+1e-8)

        x = torch.tensor(signal,dtype=torch.float32)
        y = torch.tensor(self.labels[idx],dtype=torch.long)

        return x,y


train_loader = DataLoader(
    ECGDataset(X_train,y_train),
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    ECGDataset(X_test,y_test),
    batch_size=16
)

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
        x = self.pool(self.relu(self.conv3(x)))

        x = self.global_pool(x)

        x = x.squeeze(-1)

        x = self.dropout(x)

        x = self.fc(x)

        return x


model = ECGModel().to(device)

# 🔥 Class imbalance fix
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0,1.2,1.2]).to(device)
)

# 🔥 Daha stabil learning rate
optimizer = optim.Adam(model.parameters(),lr=0.0003)

# -----------------------------
# TRAIN
# -----------------------------
epochs = 40

for epoch in range(epochs):

    torch.mps.empty_cache()

    model.train()
    total_loss = 0

    for x,y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred,y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # 🔥 DOĞRU LOSS
    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")

# -----------------------------
# TEST
# -----------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for x,y in test_loader:

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        _,predicted = torch.max(pred,1)

        total += y.size(0)
        correct += (predicted == y).sum().item()

accuracy = correct / total

print("🔥 Test Accuracy:", accuracy)

torch.save(model.state_dict(),"ecg_final_safe.pth")