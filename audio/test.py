import time
from load_dataset import AudioDataset
dataset = AudioDataset("data/audio_preprocessed/")
start = time.time()
for i in range(10):
    waveform, label = dataset[i]
end = time.time()
print("Tiempo promedio por archivo:", (end-start)/10)
