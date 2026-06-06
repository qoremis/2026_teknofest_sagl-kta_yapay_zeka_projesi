import wfdb
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Mimari Deposundan İçe Aktar
from models_ablation import ResNetECGModel, InceptionECGModel, DenseNetECGModel

dataset_root = Path(__file__).resolve().parents[1]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class_names = ["Normal", "AFIB", "AFL", "LBBB", "RBBB"]

def preprocess_signal_safe(signal, fs_original=500, fs_target=100):
    n = signal.shape[-1]
    fft_vals = np.fft.rfft(signal, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1/fs_original)
    mask = (freqs >= 0.05) & (freqs <= 45.0)
    fft_vals[:, ~mask] = 0
    filtered = np.fft.irfft(fft_vals, n=n, axis=-1)[:, ::fs_original // fs_target]
    
    normalized = np.zeros_like(filtered)
    for i in range(filtered.shape[0]):
        lead_std = np.std(filtered[i])
        if lead_std > 1e-4: normalized[i] = (filtered[i] - np.mean(filtered[i])) / lead_std
    return normalized

def extract_psd_features_raw(signal):
    return np.mean(np.abs(np.fft.rfft(signal, axis=-1)) ** 2, axis=-1)

def predict_and_plot():
    # =========================================================================
    # 🎯 TEST EDİLECEK MODELİ SEÇ ("ResNet", "Inception", "DenseNet")
    # =========================================================================
    model_name = ""
    
    if model_name == "ResNet":
        model = ResNetECGModel()
    elif model_name == "Inception":
        model = InceptionECGModel()
    elif model_name == "DenseNet":
        model = DenseNetECGModel()
        
    model_weight_path = dataset_root / "kodlar" / f"teknofest_ecg_{model_name.lower()}.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model = model.to(device).eval()
    print(f"🎯 {model_name} Başarıyla Yüklendi!\n")
    # =========================================================================

    df = pd.read_csv(dataset_root / 'test_public.csv')
    for _ in range(5):
        row = df.sample(n=1).iloc[0]
        base = str((dataset_root / row['header_path']).with_suffix(''))
        try:
            sig = preprocess_signal_safe(wfdb.rdrecord(base).p_signal.T)
            psd = extract_psd_features_raw(sig)
            
            with torch.no_grad():
                out = model(torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(device),
                            torch.tensor(psd, dtype=torch.float32).unsqueeze(0).to(device))
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred_idx = torch.argmax(out, dim=1).item()

            actual = row['label'] if 'label' in row else class_names[int(row['class_id'])]
            
            plt.figure(figsize=(10, 4))
            plt.plot(sig[1][:400], color="navy")
            plt.title(f"[{model_name}] Gerçek: {actual} | Tahmin: {class_names[pred_idx]} (%{probs[pred_idx]*100:.1f})")
            plt.grid(True)
            plt.show()
            break
        except Exception as e: continue

if __name__ == "__main__":
    predict_and_plot()