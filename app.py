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

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load('best_resnet_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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


st.title("Audio Classification using Spectrogram")
uploaded_file = st.file_uploader("Upload an audio file for classification", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    spectrogram_image = audio_to_spectrogram(uploaded_file)
    st.image(spectrogram_image, caption="Generated Spectrogram", use_column_width=True)
    
    image_tensor = transform(spectrogram_image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, pred = torch.max(output, 1)
    
    class_names = ['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4'] 
    predicted_class = class_names[pred.item()]
    
    st.write(f"**Prediction:** {predicted_class}")
