import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import tensorflow as tf
from pydub.utils import mediainfo
import os
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load Pretrained Model for Spectrogram Classification
@st.cache_resource
def load_resnet_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load('best_resnet_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

resnet_model = load_resnet_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load TensorFlow Model for Time-Based Classification
def load_model():
    return tf.keras.models.load_model("audio_classifier.h5")

model = load_model()

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Not Picked Up', 'Picked Up'])

# Audio preprocessing for Spectrogram
def audio_to_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

# Audio preprocessing for Time-Based Classification
def preprocess_audio(file_path, sr=16000, n_mels=128, duration=2.0, img_size=128):
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = np.resize(mel_spec_db, (img_size, img_size))
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        mel_spec_db = np.repeat(mel_spec_db, 3, axis=-1)
        mel_spec_db = mel_spec_db / 255.0
        return np.expand_dims(mel_spec_db, axis=0)
    except Exception as e:
        print(f"Error in preprocessing {file_path}: {e}")
        return None

def get_audio_length(file_path):
    info = mediainfo(file_path)
    return float(info['duration']) if info['duration'] != "N/A" else 0

def predict_audio(file_path):
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    duration = get_audio_length(file_path)
    if duration < 32:
        return "Not Picked Up"
    elif duration >= 61:
        return "Picked Up"
    else:
        input_data = preprocess_audio(file_path)
        if input_data is None:
            return "Error in processing audio"
        prediction = model.predict(input_data)
        return label_encoder.inverse_transform([np.argmax(prediction)])[0]

# Streamlit app layout
st.title('Audio Classification App')
tabs = st.tabs(["Audio Classification", "Time-Based Classification"])

with tabs[0]:
    st.header("Audio Classification using Spectrogram")
    uploaded_file = st.file_uploader("Upload an audio file for classification", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        spectrogram_image = audio_to_spectrogram(uploaded_file)

        st.audio(uploaded_file, format="audio/wav")

        st.image(spectrogram_image, caption="Generated Spectrogram", use_container_width=True)

        image_tensor = transform(spectrogram_image).unsqueeze(0)

        with torch.no_grad():
            output = resnet_model(image_tensor)
            _, pred = torch.max(output, 1)

        class_names = ['CALLER_TUNE/BUSY/PHONE_RINGING', 'BLANK_CALL/NOT_ENOUGH_INFO/BACKGROUND_NOISE/OTHER', 'UNAVAILABLE/OUT_OF_COVERAGE/SWITCHED_OFF/BEEP', 'NOT_VALID/TEMP_OUT_OF_SERVICE']
        predicted_class = class_names[pred.item()]

        st.write(f"**Prediction:** {predicted_class}")

with tabs[1]:
    st.header("Time-Based Classification")
    uploaded_file_time = st.file_uploader("Choose an audio file for time-based classification", type=["wav", "mp3"])

    if uploaded_file_time is not None:
        file_path = os.path.join("temp", uploaded_file_time.name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(uploaded_file_time.getbuffer())

        st.audio(uploaded_file_time, format="audio/wav")

        predicted_label = predict_audio(file_path)

        st.write(f"Predicted Label: {predicted_label}")
