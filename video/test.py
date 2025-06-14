from pathlib import Path
import numpy as np

data_root = "data/video_preprocessed"
sessions = ["Session1", "Session2", "Session3", "Session4"]
genders = ["Male", "Female"]

for session in sessions:
    for gender in genders:
        session_path = Path(data_root) / session / gender
        if session_path.exists():
            for npy_file in session_path.glob("*.npy"):
                try:
                    data = np.load(npy_file, allow_pickle=True).item()
                    frames = data["frames"]
                    emotion = data["emotion"]
                    print(f"OK: {npy_file}, frames shape: {frames.shape}, emotion: {emotion}")
                except Exception as e:
                    print(f"ERROR: {npy_file}, {str(e)}")