import os
import glob
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTModel
import streamlit as st


# 
# KONFIGURACJA STRONY I PARAMETRY GLOBALNE
# 

st.set_page_config(page_title="Wizualizacja ICBHI", layout="wide")
st.title("Wizualizacja klasyfikacji i segmentacji dźwięków oddechowych")

SEG_SR = 4000
CLF_SR = 16000

HOP_LENGTH_SEG = 100
N_MELS_SEG = 64

CLF_SEG_LEN_SEC = 5
CLF_SEG_TARGET_LEN = CLF_SR * CLF_SEG_LEN_SEC

CLASS_NAMES = ["Normal", "Crackles", "Wheezes", "Both"]

CLASS_COLORS = {
    "Normal": "lightgreen",
    "Crackles": "orange",
    "Wheezes": "skyblue",
    "Both": "salmon"
}

# Ścieżki do danych i modeli
DATASET_DIR = "C:/Users/Admin/Desktop/PracaMGR/data/audio_and_txt_files"
CRNN_MODEL_PATH = "C:/Users/Admin/Desktop/PracaMGR/models/best_crnn_model.pth"
AST_MODEL_PATH = "C:/Users/Admin/Desktop/PracaMGR/models/best_ast_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DEFINICJE ARCHITEKTUR MODELI

class RespiratorySegmentationCRNN(nn.Module):
    def __init__(self, input_channels=1, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )

        self.rnn = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.cnn(x)
        b, c, m, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * m)
        x, _ = self.rnn(x)
        return self.classifier(x)


class LungSoundAST(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out = self.ast(x)
        cls_tok = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_tok)


# FUNKCJE POMOCNICZE I ŁADOWANIE MODELI

@st.cache_resource
def load_models():
    ast_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    # CRNN
    crnn_model = RespiratorySegmentationCRNN().to(device)
    if os.path.exists(CRNN_MODEL_PATH):
        sd = torch.load(CRNN_MODEL_PATH, map_location=device, weights_only=False)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        crnn_model.load_state_dict(sd)
    else:
        raise FileNotFoundError(f"Nie znaleziono modelu CRNN:\n{CRNN_MODEL_PATH}")
    crnn_model.eval()

    # AST
    ast_model = LungSoundAST(num_classes=4).to(device)
    if os.path.exists(AST_MODEL_PATH):
        checkpoint = torch.load(AST_MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            sd = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
        else:
            sd = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        ast_model.load_state_dict(sd)
    else:
        raise FileNotFoundError(f"Nie znaleziono modelu AST:\n{AST_MODEL_PATH}")
    ast_model.eval()

    return crnn_model, ast_model, ast_extractor


@st.cache_data
def get_test_files(dataset_dir):
    txt_files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    if not txt_files:
        return []

    # Identyfikatory pacjentów i podział
    patient_ids = sorted(list(set([os.path.basename(f).split("_")[0] for f in txt_files])))
    _, test_patients = train_test_split(patient_ids, test_size=0.2, random_state=42)

    return [
        f for f in glob.glob(os.path.join(dataset_dir, "*.wav"))
        if os.path.basename(f).split("_")[0] in test_patients
    ]


def butter_highpass_filter(data, cutoff=50.0, fs=4000, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype="high", analog=False)
    return signal.filtfilt(b, a, data)


def encode_label(c, w):
    return int(c) * 1 + int(w) * 2


def load_full_recording(wav_path):
    txt_path = wav_path.replace(".wav", ".txt")

    # Dane dla CRNN
    audio_seg, _ = librosa.load(wav_path, sr=SEG_SR)
    mel = librosa.feature.melspectrogram(y=audio_seg, sr=SEG_SR, n_mels=N_MELS_SEG, hop_length=HOP_LENGTH_SEG)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    # Dane dla AST
    audio_clf, _ = librosa.load(wav_path, sr=CLF_SR)
    audio_clf_clean = butter_highpass_filter(audio_clf, fs=CLF_SR)

    # Ground Truth
    gt_data = []
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    start, end = float(parts[0]), float(parts[1])
                    c, w = int(parts[2]), int(parts[3])
                    label_name = CLASS_NAMES[encode_label(c, w)]
                    gt_data.append((start, end, label_name))
                elif len(parts) >= 2:
                    gt_data.append((float(parts[0]), float(parts[1]), "Normal"))

    return mel_db, gt_data, audio_clf_clean


def predict_segment(model, extractor, segment):
    if len(segment) < int(CLF_SR * 0.1):
        return "Normal"

    # Przycięcie lub dopełnienie do 5 sekund
    if len(segment) > CLF_SEG_TARGET_LEN:
        segment = segment[:CLF_SEG_TARGET_LEN]
    elif len(segment) < CLF_SEG_TARGET_LEN:
        segment = np.pad(segment, (0, CLF_SEG_TARGET_LEN - len(segment)))

    inputs = extractor(segment, sampling_rate=CLF_SR, padding="max_length", return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values)
        probs = torch.softmax(logits, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()

    return CLASS_NAMES[pred_class_idx]


# INTERFEJS INTERAKTYWNY (STREAMLIT)

st.sidebar.header("Konfiguracja")

crnn_model, ast_model, ast_extractor = load_models()
test_files = get_test_files(DATASET_DIR)

if not test_files:
    st.error(f"Nie znaleziono plików testowych. Upewnij się, że folder `{DATASET_DIR}` zawiera pliki .wav i .txt.")
else:
    file_options = {os.path.basename(f): f for f in test_files}
    selected_file_name = st.sidebar.selectbox("Wybierz nagranie testowe pacjenta:", list(file_options.keys()))
    selected_wav_path = file_options[selected_file_name]

    if st.sidebar.button("Uruchom analizę"):
        with st.spinner("Przetwarzanie nagrania... to może potrwać kilka sekund."):
            
            # 1. Wczytanie
            mel_db_seg, gt_data, audio_clf_clean = load_full_recording(selected_wav_path)

            # 2. Predykcja CRNN
            with torch.no_grad():
                crnn_input = torch.tensor(mel_db_seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                probs = torch.sigmoid(crnn_model(crnn_input)).squeeze().cpu().numpy()

            time_axis = np.arange(len(probs)) * (HOP_LENGTH_SEG / SEG_SR)
            probs_smooth = np.convolve(probs, np.ones(5) / 5, mode="same")

            # Detekcja granic CRNN (height = 0.07, distance = 40)
            peaks, _ = find_peaks(probs_smooth, height=0.07, distance=40)

            # 3. Wyznaczenie przedziałów
            pred_intervals = []
            
            # Początek nagrania -> pierwsza granica
            if len(peaks) > 0 and peaks[0] > (2.0 * SEG_SR / HOP_LENGTH_SEG):
                pred_intervals.append((0.0, time_axis[peaks[0]]))

            # Przedziały pomiędzy kolejnymi granicami
            for i in range(len(peaks) - 1):
                pred_intervals.append((time_axis[peaks[i]], time_axis[peaks[i + 1]]))

            # Ostatnia granica -> koniec nagrania
            if len(peaks) > 0 and (len(probs) - peaks[-1]) > (2.0 * SEG_SR / HOP_LENGTH_SEG):
                pred_intervals.append((time_axis[peaks[-1]], time_axis[-1]))

            # 4. Wykres
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 6), sharex=True)

            # Subplot 1: Ground Truth
            for s, e, tl in gt_data:
                ax1.add_patch(
                    patches.Rectangle((s, 0), e - s, 1, edgecolor="black", facecolor=CLASS_COLORS.get(tl, "white"), alpha=0.8)
                )
                ax1.text(s + (e - s) / 2, 0.5, tl, ha="center", va="center", fontsize=8, fontweight="bold")

            ax1.set_ylim(0, 1)
            ax1.set_yticks([])
            ax1.set_title("Ground Truth (Oznaczenia oryginalne)", fontsize=10, fontweight="bold")

            # Subplot 2: CRNN
            ax2.plot(time_axis, probs_smooth, color="blue")
            for s, e in pred_intervals:
                ax2.add_patch(
                    patches.Rectangle((s, 0), e - s, 1, edgecolor="black", facecolor="red", alpha=0.3)
                )

            ax2.axhline(y=0.07, linestyle="--", linewidth=1, label="Próg detekcji = 0.07")
            ax2.set_ylim(0, 1)
            ax2.set_yticks([])
            ax2.set_title("CRNN - Prawdopodobieństwo krawędzi (Detekcja)", fontsize=10, fontweight="bold")
            ax2.legend(loc="upper right")

            # Subplot 3: AST
            for s, e in pred_intervals:
                seg_audio = audio_clf_clean[int(s * CLF_SR):int(e * CLF_SR)]
                pred_label = predict_segment(ast_model, ast_extractor, seg_audio)

                ax3.add_patch(
                    patches.Rectangle((s, 0), e - s, 1, edgecolor="black", facecolor=CLASS_COLORS.get(pred_label, "white"), alpha=0.8)
                )
                ax3.text(s + (e - s) / 2, 0.5, pred_label, ha="center", va="center", fontsize=8, fontweight="bold")

            ax3.set_ylim(0, 1)
            ax3.set_yticks([])
            ax3.set_xlabel("Czas [s]")
            ax3.set_title("AST Transformer - Klasyfikacja cykli", fontsize=10, fontweight="bold")

            plt.suptitle(f"Analiza nagrania: {selected_file_name}", fontweight="bold", fontsize=14)
            plt.tight_layout()

            st.pyplot(fig)