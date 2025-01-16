import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load the pre-trained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Load the audio file
audio_path = "/sample-data/speech.mp3"
audio, sample_rate = librosa.load(audio_path, sr=16000)

# Process the audio
inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)

# Extract audio features using the model
with torch.no_grad():
    audio_features = model(**inputs).last_hidden_state

# Convert the audio features to a vector
audio_vector = audio_features.mean(dim=1).squeeze().numpy()

print("Audio vector:", audio_vector)
