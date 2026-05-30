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

# =====================================================================
# BÖLÜM 1: YOL VE CİHAZ AYARLARI
# =====================================================================
dataset_root = Path(__file__).resolve().parents[1]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️ Aktif Eğitim Cihazı: {device}")

# =====================================================================
# BÖLÜM 2: KANAL BAZLI GÜVENLİ SİNYAL İŞLEME (NaN ENGELLEYİCİ)
# =====================================================================
def preprocess_signal_safe(signal, fs_original=500, fs_target=100):
    n = signal.shape[-1]
    fft_vals = np.fft.rfft(signal, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1/fs_original)
    
    mask = (freqs >= 0.05) & (freqs <= 45.0)
    fft_vals[:, ~mask] = 0
    filtered_signal = np.fft.irfft(fft_vals, n=n, axis=-1)
    
    step = fs_original // fs_target
    downsampled_signal = filtered_signal[:, ::step]
    
    # 🚨 KRİTİK GÜNCELLEME: Her kanalı kendi içinde güvenli normalize et
    normalized_signal = np.zeros_like(downsampled_signal)
    for i in range(downsampled_signal.shape[0]):
        lead_std = np.std(downsampled_signal[i])
        if lead_std > 1e-4: # Sinyal düz çizgi değilse normalize et
            normalized_signal[i] = (downsampled_signal[i] - np.mean(downsampled_signal[i])) / lead_std
        else:
            normalized_signal[i] = np.zeros_like(downsampled_signal[i]) # Ölü kanalı sıfırla
            
    return normalized_signal

def extract_psd_features_raw(signal):
    """PSD hesaplamasını ham bırakıp normalizasyonu PyTorch BatchNorm'a devrediyoruz."""
    fft_vals = np.abs(np.fft.rfft(signal, axis=-1))
    psd_mean = np.mean(fft_vals ** 2, axis=-1)
    return psd_mean

# =====================================================================
# BÖLÜM 3: VERİ SETİ SINIFI
# =====================================================================
class TeknofestECGDataset(Dataset):
    def __init__(self, dataframe, augment=False):
        self.df = dataframe
        self.augment = augment
        self.records = []
        
        for idx, row in self.df.iterrows():
            header_path = dataset_root / row['header_path']
            record_base = str(header_path.with_suffix(''))
            try:
                record = wfdb.rdrecord(record_base)
                signal = record.p_signal.T
                
                processed_sig = preprocess_signal_safe(signal)
                psd_feat = extract_psd_features_raw(processed_sig)
                
                # Veride halihazırda NaN var mı kontrol et, varsa eğitim setine hiç alma!
                if np.isnan(processed_sig).any() or np.isnan(psd_feat).any():
                    continue
                    
                self.records.append({
                    'signal': torch.tensor(processed_sig, dtype=torch.float32),
                    'psd': torch.tensor(psd_feat, dtype=torch.float32),
                    'label': torch.tensor(row['class_id'], dtype=torch.long)
                })
            except:
                continue

    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        signal = self.records[idx]['signal']
        psd = self.records[idx]['psd']
        label = self.records[idx]['label']
        
        if self.augment and np.random.rand() > 0.5:
            shift = np.random.randint(-20, 20)
            signal = torch.roll(signal, shifts=shift, dims=-1)
            
        return signal, psd, label

# =====================================================================
# BÖLÜM 4: MATEMATİKSEL OLARAK KUSURSUZ MODEL MİMARİSİ
# =====================================================================
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
        scores = torch.bmm(q, k) / np.sqrt(x.shape[1] // 4)
        attn = self.softmax(scores)
        return x + torch.bmm(v, attn.permute(0, 2, 1))

class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.attention = SinyalAttention(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # PSD özniteliklerini matematiksel olarak kontrol altında tutacak katman
        self.psd_bn = nn.BatchNorm1d(12)
        
        # Son sınıflandırıcı öncesi stabilizasyon
        self.hybrid_norm = nn.LayerNorm(140)
        
        self.classifier = nn.Sequential(
            nn.Linear(140, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 5)
        )

    def forward(self, x, psd):
        x = self.features(x)
        x = self.attention(x)
        x = self.global_pool(x)
        x = torch.squeeze(x, dim=-1)
        
        # PSD'yi güvenli bir şekilde ölçekle
        psd_scaled = self.psd_bn(psd)
        
        # Birleştirme ve katman normalizasyonu (NaN panzehiri)
        hybrid = torch.cat((x, psd_scaled), dim=1)
        hybrid = self.hybrid_norm(hybrid)
        return self.classifier(hybrid)

# =====================================================================
# BÖLÜM 5: VERİ BÖLME VE OTOMATİK DENGELİ YÜKLEME
# =====================================================================
def prepare_data_loaders():
    train_df = pd.read_csv(dataset_root / 'train.csv')
    val_df = pd.read_csv(dataset_root / 'validation.csv')
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    labels = combined_df['class_id'].values
    computed_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(computed_weights, dtype=torch.float32).to(device)

    train_split, val_split = train_test_split(
        combined_df, test_size=0.20, random_state=42, stratify=combined_df['class_id']
    )
    
    train_loader = DataLoader(TeknofestECGDataset(train_split, augment=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(TeknofestECGDataset(val_split, augment=False), batch_size=32, shuffle=False)
    return train_loader, val_loader, class_weights

# =====================================================================
# BÖLÜM 6: KESİNTİSİZ EĞİTİM DÖNGÜSÜ
# =====================================================================
def train_pipeline():
    train_loader, val_loader, class_weights = prepare_data_loaders()
    model = ECGModel().to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    train_losses = []
    val_f1_scores = []
    
    print("\n🛡️ Güvenli (NaN Korumalı) Eğitim Başlatılıyor...")
    for epoch in range(40):
        model.train()
        total_loss = 0
        skipped_batches = 0
        
        for signals, psds, labels in train_loader:
            signals, psds, labels = signals.to(device), psds.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals, psds)
            loss = criterion(outputs, labels)
            
            # Aşırı güvenlik tedbiri
            if torch.isnan(loss):
                skipped_batches += 1
                continue
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()

        # Geçerli batch sayısına bölerek ortalama alıyoruz
        effective_batches = len(train_loader) - skipped_batches
        epoch_loss = total_loss / effective_batches if effective_batches > 0 else 0
        train_losses.append(epoch_loss)

        # Doğrulama
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for signals, psds, labels in val_loader:
                outputs = model(signals.to(device), psds.to(device))
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        val_f1_scores.append(macro_f1)
        
        skip_msg = f" (Atlanan Batch: {skipped_batches})" if skipped_batches > 0 else ""
        print(f"Epoch [{epoch+1}/40] - Loss: {epoch_loss:.4f} - Val Macro F1: {macro_f1:.4f}{skip_msg}")

    torch.save(model.state_dict(), Path(__file__).resolve().parent / "teknofest_ecg_model.pth")
    print(f"\n✅ Kararlı model başarıyla kaydedildi.")

    # Grafikleri Çizdirme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(range(1, 41), train_losses, marker='o', color='purple', label='Loss')
    ax1.set_title('Sayısal Kararlı Loss Eğrisi')
    ax1.grid(True)
    
    ax2.plot(range(1, 41), val_f1_scores, marker='s', color='teal', label='Macro F1')
    ax2.set_title('Doğrulama Başarısı')
    ax2.grid(True)
    plt.show()

if __name__ == "__main__":
    train_pipeline()