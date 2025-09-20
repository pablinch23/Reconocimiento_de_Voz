import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define the directory where your audio files are located
AUDIO_DIR = "audios_dataset"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)
    print(f"Directory '{AUDIO_DIR}' created. Please place your audio files here.")

# Función para cargar audio, segmentar y extraer características con clasificación adaptativa
def process_audio_file(file_path):

    y, sr = librosa.load(file_path, sr=16000)
    segment_length = 0.1 # 100 ms segments
    hop_length = int(segment_length * sr)
    frame_length = hop_length

    # Extract acoustic features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length, n_fft=frame_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length, frame_length=frame_length)[0]
    energy = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=frame_length)[0]

    # Clasificación adaptativa (S/U/V)
    energy_threshold = np.mean(energy) * 0.5  # Silence threshold
    zcr_threshold = np.mean(zcr) + np.std(zcr) * 0.2  # Unvoiced threshold
    
    labels = []
    for i in range(len(zcr)):
        if energy[i] < energy_threshold:
            labels.append("S")  # Silence
        elif zcr[i] > zcr_threshold:
            labels.append("U")  # Unvoiced
        else:
            labels.append("V")  # Voiced

    return mfccs.T, labels, sr, y, hop_length

# 2. Function to plot all required visualizations for a single audio file
def plot_audio_visualizations(y, sr, labels, hop_length, file_name):
    """
    Generates and displays a combined plot with a segmented waveform,
    a full waveform, and a narrowband spectrogram.
    """
    colors = {"S": "lightgray", "U": "red", "V": "green"}
    label_colors = [colors[label] for label in labels]

    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: Segmented Waveform with S/U/V labels 
    librosa.display.waveshow(y, sr=sr, color="black", alpha=0.5, ax=axs[0])
    for j, label in enumerate(labels):
        start = j * hop_length / sr
        end = (j + 1) * hop_length / sr
        axs[0].axvspan(start, end, color=colors[label], alpha=0.3)
        axs[0].text((start + end) / 2, 0.8 * np.max(y), label, fontsize=8, ha="center", va="top")
    axs[0].set_title(f"Audio: {file_name} - Segmentación (100 ms)")
    axs[0].set_xlabel("Tiempo (s)")
    axs[0].set_ylabel("Amplitud")

    # Plot 2: Full Waveform in time domain [cite: 730]
    librosa.display.waveshow(y, sr=sr, color="black", ax=axs[1])
    axs[1].set_title(f"Audio: {file_name} - Señal completa en dominio del tiempo")
    axs[1].set_xlabel("Tiempo (s)")
    axs[1].set_ylabel("Amplitud")

    # Plot 3: Narrowband Spectrogram in frequency domain [cite: 731]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, hop_length=512, x_axis="time", y_axis="hz", cmap="magma", ax=axs[2])
    axs[2].set_title(f"Audio: {file_name} - Espectrograma de banda estrecha")
    fig.colorbar(img, ax=axs[2], format="%+2.0f dB")

    plt.tight_layout()
    plt.show()

# 3. Function for classification and plotting confusion matrices
def run_classification_models(X, y):
   
    X = np.array(X)
    y = np.array(y)

    # Normalización de las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División del dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Inicialización de los modelos
    classifiers = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["S", "U", "V"]
    
    for ax, (name, classifier) in zip(axs, classifiers.items()):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_title(f"Matriz de Confusión\n{name}")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
                
        print(f"\n===== Resultados {name} =====")
        print(classification_report(y_test, y_pred, target_names=labels, labels=labels))

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    features_list = []
    labels_list = []
    
    # Process each audio file in the specified directory
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])
    for file_name in audio_files:
        file_path = os.path.join(AUDIO_DIR, file_name)
        features, labels, sr, y, hop_length = process_audio_file(file_path)
        features_list.extend(features)
        labels_list.extend(labels)
        
        # Plot visualizations for each audio
        plot_audio_visualizations(y, sr, labels, hop_length, file_name)

    # Run the classification and evaluation
    run_classification_models(features_list, labels_list)