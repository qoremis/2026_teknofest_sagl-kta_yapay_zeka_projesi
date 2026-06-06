import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

# Mimari Deposundan Modelleri İçe Aktar
from models_ablation import ResNetECGModel, InceptionECGModel, DenseNetECGModel

dataset_root = Path(__file__).resolve().parents[1]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def preprocess_signal_safe(signal, fs_original=500, fs_target=100):
    n = signal.shape[-1]
    fft_vals = np.fft.rfft(signal, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1/fs_original)
    mask = (freqs >= 0.05) & (freqs <= 45.0)
    fft_vals[:, ~mask] = 0
    filtered_signal = np.fft.irfft(fft_vals, n=n, axis=-1)
    step = fs_original // fs_target
    downsampled = filtered_signal[:, ::step]
    
    normalized = np.zeros_like(downsampled)
    for i in range(downsampled.shape[0]):
        lead_std = np.std(downsampled[i])
        if lead_std > 1e-4:
            normalized[i] = (downsampled[i] - np.mean(downsampled[i])) / lead_std
    return normalized

def extract_psd_features_raw(signal):
    return np.mean(np.abs(np.fft.rfft(signal, axis=-1)) ** 2, axis=-1)

class TeknofestECGDataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.df = dataframe
        self.augment = augment
        self.records = []
        for _, row in self.df.iterrows():
            record_base = str((dataset_root / row['header_path']).with_suffix(''))
            try:
                sig = wfdb.rdrecord(record_base).p_signal.T
                proc_sig = preprocess_signal_safe(sig)
                psd = extract_psd_features_raw(proc_sig)
                if np.isnan(proc_sig).any() or np.isnan(psd).any(): continue
                self.records.append({
                    'signal': torch.tensor(proc_sig, dtype=torch.float32),
                    'psd': torch.tensor(psd, dtype=torch.float32),
                    'label': torch.tensor(row['class_id'], dtype=torch.long)
                })
            except: continue
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        return self.records[idx]['signal'], self.records[idx]['psd'], self.records[idx]['label']

def train_pipeline():
    # Veri Yükleme
    df = pd.concat([pd.read_csv(dataset_root / 'train.csv'), pd.read_csv(dataset_root / 'validation.csv')], ignore_index=True)
    labels = df['class_id'].values
    weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(labels), y=labels), dtype=torch.float32).to(device)
    train_split, val_split = train_test_split(df, test_size=0.20, random_state=42, stratify=labels)
    
    train_loader = DataLoader(TeknofestECGDataset(train_split, augment=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(TeknofestECGDataset(val_split, augment=False), batch_size=32, shuffle=False)
    
    # =========================================================================
    # 🎯 MODEL SEÇİM EKRANI (Ablation Study için sadece burayı değiştir)
    # =========================================================================
    model_name = "DenseNet" # Seçenekler: "ResNet", "Inception", "DenseNet"
    
    if model_name == "ResNet":
        model = ResNetECGModel().to(device)
    elif model_name == "Inception":
        model = InceptionECGModel().to(device)
    elif model_name == "DenseNet":
        model = DenseNetECGModel().to(device)
    
    print(f"\n🚀 {model_name} Tabanlı Model Eğitimi Başlıyor...")
    # =========================================================================

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    train_losses, val_f1_scores = [], []
    for epoch in range(40):
        model.train()
        total_loss, valid_batches = 0, 0
        for sigs, psds, lbls in train_loader:
            sigs, psds, lbls = sigs.to(device), psds.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sigs, psds), lbls)
            if torch.isnan(loss): continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1

        epoch_loss = total_loss / valid_batches if valid_batches > 0 else 0
        train_losses.append(epoch_loss)

        model.eval()
        all_preds, all_lbls = [], []
        with torch.no_grad():
            for sigs, psds, lbls in val_loader:
                preds = torch.argmax(model(sigs.to(device), psds.to(device)), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_lbls.extend(lbls.numpy())
        
        macro_f1 = f1_score(all_lbls, all_preds, average='macro')
        val_f1_scores.append(macro_f1)
        print(f"Epoch [{epoch+1}/40] - Loss: {epoch_loss:.4f} - Val Macro F1: {macro_f1:.4f}")

    # Eğitilen modele özel isimle kaydet
    save_path = Path(__file__).resolve().parent / f"teknofest_ecg_{model_name.lower()}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ {model_name} modeli '{save_path.name}' olarak kaydedildi.")

if __name__ == "__main__":
    train_pipeline()