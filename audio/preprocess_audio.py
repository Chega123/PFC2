import torchaudio
import os
import numpy as np
from make_dataset_audio import collect_dataset_info

save="data/audio_preprocessed/"
torchaudio.set_audio_backend("soundfile")

def preprocess_and_save():
    dataset= collect_dataset_info()
    for item in dataset:
        path = item["path"]
        emotion = item["emotion"]
        session = item["session"]
        gender = item["gender"]
        name = item["name"]
        try:
            waveform,sr=torchaudio.load(path)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq= 16000) # frecuencia q usas wav2vec
            waveform=resampler(waveform)
            waveform=waveform/waveform.abs().max() #normalizamos 
            waveform = waveform.squeeze().numpy()

            out_dir= os.path.join(save,session,gender)
            os.makedirs(out_dir,exist_ok=True)
            out_path = os.path.join(out_dir, name + ".npy")
            data_to_save = {
                "path": path,       # ruta original del .wav
                "emotion": emotion
            }
            np.save(out_path, data_to_save)

        except Exception as e:
            print(f"Error con {path}: {e}")



if __name__=="__main__":
    preprocess_and_save()