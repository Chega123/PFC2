import numpy as np

# Cargar un audio de ejemplo
audio_file = "D:/tesis/tesis/data/audio_preprocessed/Session1/Female/Ses01F_impro01_F000.npy"
audio_data = np.load(audio_file, allow_pickle=True)

print("Tipo:", type(audio_data))
print("Shape:", audio_data.shape if hasattr(audio_data, 'shape') else 'No shape')
print("Dtype:", audio_data.dtype)

if audio_data.dtype == object:
    audio_data = audio_data.item()
    print("\nDespués de .item():")
    print("Tipo:", type(audio_data))
    print("Keys:", audio_data.keys() if isinstance(audio_data, dict) else 'No es dict')
    
    waveform = audio_data.get("waveform")
    print("\nWaveform:")
    print("Tipo:", type(waveform))
    print("Shape:", waveform.shape if hasattr(waveform, 'shape') else 'No shape')