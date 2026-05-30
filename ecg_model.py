import wfdb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================================
# BÖLÜM 1: YOL VE CİHAZ AYARLARI
# =====================================================================
# ecg_model.py dosyasının bulunduğu konuma göre üst dizini bulur
dataset_root = Path(__file__).resolve().parents[1]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class_names = ["Normal", "AFIB", "AFL", "LBBB", "RBBB"]

# =====================================================================
# BÖLÜM 2: GÜVENLİ SİNYAL İŞLEME (Eğitimle Birebir Aynı)
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
    
    # Kanal bazlı güvenli normalizasyon
    normalized_signal = np.zeros_like(downsampled_signal)
    for i in range(downsampled_signal.shape[0]):
        lead_std = np.std(downsampled_signal[i])
        if lead_std > 1e-4:
            normalized_signal[i] = (downsampled_signal[i] - np.mean(downsampled_signal[i])) / lead_std
        else:
            normalized_signal[i] = np.zeros_like(downsampled_signal[i])
            
    return normalized_signal

def extract_psd_features_raw(signal):
    fft_vals = np.abs(np.fft.rfft(signal, axis=-1))
    return np.mean(fft_vals ** 2, axis=-1)

# =====================================================================
# BÖLÜM 3: EĞİTİLEN MODELİN BİREBİR AYNI MİMARİSİ (Hata Çözücü)
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
        return x + torch.bmm(v, self.softmax(scores).permute(0, 2, 1))

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
        
        # 🚨 İŞTE HATAYI ÇÖZEN KATMAN: Eksik olan BatchNorm katmanını ekledik
        self.psd_bn = nn.BatchNorm1d(12)
        
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
        
        # Hata veren parametreleri burada ileri beslemede işliyoruz
        psd_scaled = self.psd_bn(psd)
        
        hybrid = torch.cat((x, psd_scaled), dim=1)
        hybrid = self.hybrid_norm(hybrid)
        return self.classifier(hybrid)

# =====================================================================
# BÖLÜM 4: ÇIKARIM VE ANALİZ MODÜLÜ
# =====================================================================
def predict_and_plot():
    test_csv_path = dataset_root / 'test_public.csv'
    df = pd.read_csv(test_csv_path)
    
    # Model ağırlıklarının yükleneceği dosya yolu
    model_weight_path = dataset_root / "kodlar" / "teknofest_ecg_model.pth"
    
    # Model ayağa kaldırılıyor ve ağırlıklar yükleniyor
    model = ECGModel()
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model = model.to(device).eval()

    print("🎯 Model başarıyla yüklendi! Canlı EKG analizleri başlatılıyor...\n")

    for _ in range(5): # Rastgele 5 tane kaydı analiz edip ekrana çizdirir
        random_row = df.sample(n=1).iloc[0]
        record_base = str((dataset_root / random_row['header_path']).with_suffix(''))
        
        try:
            record = wfdb.rdrecord(record_base)
            processed_sig = preprocess_signal_safe(record.p_signal.T)
            psd_feat = extract_psd_features_raw(processed_sig)
            
            input_tensor = torch.tensor(processed_sig, dtype=torch.float32).unsqueeze(0).to(device)
            psd_tensor = torch.tensor(psd_feat, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor, psd_tensor)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                predicted_idx = torch.argmax(output, dim=1).item()

            actual_label = random_row['label'] if 'label' in random_row else class_names[int(random_row['class_id'])]
            
            plt.figure(figsize=(12, 4))
            plt.plot(processed_sig[1][:400], color="purple", label="Derivasyon II (Filtrelenmiş)")
            
            is_correct = actual_label == class_names[predicted_idx]
            title_color = "darkgreen" if is_correct else "crimson"
            
            plt.title(f"Kayıt ID: {random_row['record_id']}\n"
                      f"Gerçek Durum: {actual_label} | Model Tahmini: {class_names[predicted_idx]} (%{probabilities[predicted_idx]*100:.1f})", 
                      fontsize=12, fontweight='bold', color=title_color)
            plt.xlabel("Zaman Adımları (100 Hz)")
            plt.ylabel("Genlik")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
            break
        except Exception as e:
            continue

if __name__ == "__main__":
    predict_and_plot()