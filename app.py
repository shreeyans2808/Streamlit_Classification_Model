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

# Image Transform for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to convert audio to spectrogram image
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

# UI: Title and file uploader
st.title('Audio Classification using Spectrogram')
uploaded_files = st.file_uploader("Upload audio files (WAV, MP3, OGG, FLAC)", type=["wav", "mp3", "ogg", "flac"], accept_multiple_files=True)

# Class names for predictions
class_names = [
    'CALLER_TUNE/BUSY/PHONE_RINGING',
    'BLANK_CALL/NOT_ENOUGH_INFO/BACKGROUND_NOISE/OTHER',
    'UNAVAILABLE/OUT_OF_COVERAGE/SWITCHED_OFF/BEEP',
    'NOT_VALID/TEMP_OUT_OF_SERVICE'
]

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"### 🔊 File: `{uploaded_file.name}`")
        st.audio(uploaded_file, format="audio/wav")

        # Generate spectrogram and show
        spectrogram_image = audio_to_spectrogram(uploaded_file)
        st.image(spectrogram_image, caption="Spectrogram", use_container_width=True)

        # Predict using the model
        image_tensor = transform(spectrogram_image).unsqueeze(0)
        with torch.no_grad():
            output = resnet_model(image_tensor)
            _, pred = torch.max(output, 1)

        predicted_class = class_names[pred.item()]
        st.write(f"**Prediction:** {predicted_class}")
        st.markdown("---")
