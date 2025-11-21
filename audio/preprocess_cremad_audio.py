import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class CREMADAudioPreprocessor:
    def __init__(
        self,
        cremad_root: str,
        output_dir: str,
        audio_format: str = 'wav'
    ):
        self.cremad_root = Path(cremad_root)
        self.output_dir = Path(output_dir)
        self.audio_format = audio_format
        
        # Mapeo de emociones CREMA-D a formato estándar (minúsculas como en tu código)
        self.emotion_mapping = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fear",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad"
        }
        
        # Las 12 frases del dataset
        self.sentences = {
            "IEO": "It's eleven o'clock",
            "TIE": "That is exactly what happened",
            "IOM": "I'm on my way to the meeting",
            "IWW": "I wonder what this is about",
            "TAI": "The airplane is almost full",
            "MTI": "Maybe tomorrow it will be cold",
            "IWL": "I would like a new alarm clock",
            "ITH": "I think I have a doctor's appointment",
            "DFA": "Don't forget a jacket",
            "ITS": "I think I've seen this before",
            "TSI": "The surface is slick",
            "WSI": "We'll stop in a couple of minutes"
        }
        
        # Cargar demografía
        demographics_path = self.cremad_root / "VideoDemographics.csv"
        if demographics_path.exists():
            self.demographics = pd.read_csv(demographics_path)
            print(f"Demografía cargada: {len(self.demographics)} actores")
        else:
            print("Advertencia: No se encontró VideoDemographics.csv")
            self.demographics = None

    def parse_filename(self, filename: str):
        """
        Parsea el nombre de archivo CREMA-D
        Formato: ActorID_Sentence_Emotion_Level.ext
        Ejemplo: 1001_DFA_ANG_XX.wav
        """
        # Remover extensión
        name = filename.replace('.wav', '').replace('.mp3', '').replace('.flv', '')
        parts = name.split('_')
        
        if len(parts) != 4:
            return None
        
        actor_id, sentence_code, emotion_code, level = parts
        
        # Verificar que el código de emoción sea válido
        if emotion_code not in self.emotion_mapping:
            return None
        
        return {
            'actor_id': actor_id,
            'sentence_code': sentence_code,
            'sentence': self.sentences.get(sentence_code, ""),
            'emotion': self.emotion_mapping[emotion_code],
            'emotion_code': emotion_code,
            'level': level,
            'filename': filename
        }

    def get_actor_info(self, actor_id: str):

        if self.demographics is None:
            return {"gender": "Unknown", "age": None, "race": None}
        
        actor_row = self.demographics[self.demographics['ActorID'] == int(actor_id)]
        if len(actor_row) > 0:
            info = actor_row.iloc[0]
            return {
                "gender": "Male" if info['Sex'] == "Male" else "Female",
                "age": info['Age'],
                "race": info['Race']
            }
        return {"gender": "Unknown", "age": None, "race": None}

    def split_by_actors(self, file_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Divide el dataset por actores para evitar data leakage
        Similar a como se hace con sesiones en IEMOCAP
        """
        # Obtener lista única de actores
        actors = list(set([self.parse_filename(f)['actor_id'] for f in file_list]))
        print(f"Total de actores únicos: {len(actors)}")
        
        # Primera división: train vs (val+test)
        train_actors, val_test_actors = train_test_split(
            actors, 
            test_size=(val_ratio + test_ratio), 
            random_state=random_state
        )
        
        # Segunda división: val vs test
        relative_test_size = test_ratio / (val_ratio + test_ratio)
        val_actors, test_actors = train_test_split(
            val_test_actors, 
            test_size=relative_test_size, 
            random_state=random_state
        )
        
        # Asignar archivos a splits
        train_files = [f for f in file_list if self.parse_filename(f)['actor_id'] in train_actors]
        val_files = [f for f in file_list if self.parse_filename(f)['actor_id'] in val_actors]
        test_files = [f for f in file_list if self.parse_filename(f)['actor_id'] in test_actors]
        
        print(f"\n📊 División del dataset:")
        print(f"  Train: {len(train_actors)} actores ({len(train_files)} archivos)")
        print(f"  Val:   {len(val_actors)} actores ({len(val_files)} archivos)")
        print(f"  Test:  {len(test_actors)} actores ({len(test_files)} archivos)")
        
        return {
            'Session1': train_files,  # Sesión 1 = train
            'Session5': val_files,    # Sesión 5 = validation (como en tu código)
            'Session6': test_files    # Sesión 6 = test
        }

    def process_dataset(self):
        """
        Procesa todo el dataset CREMA-D y genera archivos .npy
        """
        # Determinar directorio de audio
        if self.audio_format == 'wav':
            audio_dir = self.cremad_root / "AudioWAV"
        elif self.audio_format == 'mp3':
            audio_dir = self.cremad_root / "AudioMP3"
        else:
            raise ValueError(f"Formato no soportado: {self.audio_format}")
        
        if not audio_dir.exists():
            raise FileNotFoundError(f"No se encontró el directorio: {audio_dir}")
        
        print(f"Buscando archivos en: {audio_dir}")
        
        # Obtener todos los archivos de audio
        audio_files = list(audio_dir.glob(f"*.{self.audio_format}"))
        print(f"Archivos encontrados: {len(audio_files)}")
        
        # Filtrar y parsear archivos válidos
        valid_files = []
        emotion_counts = {}
        
        for audio_file in audio_files:
            parsed = self.parse_filename(audio_file.name)
            if parsed is not None:
                valid_files.append(audio_file.name)
                emotion = parsed['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"✓ Archivos válidos: {len(valid_files)}")
        print(f"\nDistribución de emociones:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion:10s}: {count:4d} archivos")
        
        # Dividir dataset por actores
        splits = self.split_by_actors(valid_files)
        
        # Procesar cada split
        total_processed = 0
        for session_name, file_list in splits.items():
            print(f"\n{'='*60}")
            print(f"Procesando {session_name} ({len(file_list)} archivos)...")
            print(f"{'='*60}")
            
            for audio_filename in tqdm(file_list, desc=f"Procesando {session_name}"):
                parsed = self.parse_filename(audio_filename)
                if parsed is None:
                    continue
                
                # Ruta completa al archivo de audio
                audio_path = audio_dir / audio_filename
                
                if not audio_path.exists():
                    print(f"⚠ Archivo no encontrado: {audio_path}")
                    continue
                
                # Obtener información del actor
                actor_info = self.get_actor_info(parsed['actor_id'])
                
                # Crear el diccionario de datos (compatible con tu AudioDataset)
                data = {
                    'path': str(audio_path.absolute()),  # Ruta absoluta al WAV
                    'emotion': parsed['emotion'],         # Emoción en minúsculas
                    'sentence': parsed['sentence'],       # Transcripción
                    'actor_id': parsed['actor_id'],
                    'gender': actor_info['gender'],
                    'level': parsed['level'],
                    'emotion_code': parsed['emotion_code']
                }
                
                # Crear directorio de salida
                output_subdir = self.output_dir / session_name / actor_info['gender']
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                # Guardar archivo .npy
                output_filename = f"{parsed['actor_id']}_{parsed['sentence_code']}_{parsed['emotion_code']}.npy"
                output_path = output_subdir / output_filename
                
                np.save(output_path, data, allow_pickle=True)
                total_processed += 1
        
        print(f"\n{'='*60}")
        print(f"✅ Preprocesamiento completado!")
        print(f"   Total procesados: {total_processed} archivos")
        print(f"   Guardados en: {self.output_dir}")
        print(f"{'='*60}")
        
        # Guardar estadísticas
        self.save_statistics(splits)
        
        return total_processed

    def save_statistics(self, splits):
        """Guarda estadísticas del dataset procesado"""
        stats_path = self.output_dir / "dataset_stats.txt"
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CREMA-D Dataset - Estadísticas de Preprocesamiento\n")
            f.write("="*60 + "\n\n")
            
            f.write("MODALIDAD: AUDIO\n")
            f.write(f"Formato: {self.audio_format.upper()}\n\n")
            
            total_files = 0
            for split_name, file_list in splits.items():
                total_files += len(file_list)
                f.write(f"\n{split_name}:\n")
                f.write(f"  Total archivos: {len(file_list)}\n")
                
                # Contar emociones
                emotions = {}
                genders = {}
                for filename in file_list:
                    parsed = self.parse_filename(filename)
                    if parsed:
                        emotion = parsed['emotion']
                        emotions[emotion] = emotions.get(emotion, 0) + 1
                        
                        actor_info = self.get_actor_info(parsed['actor_id'])
                        gender = actor_info['gender']
                        genders[gender] = genders.get(gender, 0) + 1
                
                f.write("  Distribución de emociones:\n")
                for emotion, count in sorted(emotions.items()):
                    pct = (count / len(file_list)) * 100
                    f.write(f"    {emotion:10s}: {count:4d} ({pct:5.2f}%)\n")
                
                f.write("  Distribución por género:\n")
                for gender, count in sorted(genders.items()):
                    pct = (count / len(file_list)) * 100
                    f.write(f"    {gender:10s}: {count:4d} ({pct:5.2f}%)\n")
            
            f.write(f"\nTOTAL GENERAL: {total_files} archivos\n")
        
        print(f" Estadísticas guardadas en: {stats_path}")


def main():
    """Función principal para ejecutar el preprocesamiento"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocesar CREMA-D para modalidad de audio')
    parser.add_argument('--cremad_root', type=str, default='D:/tesis/dataset2/crema-d-mirror',
                        help='Ruta raíz del dataset CREMA-D')
    parser.add_argument('--output_dir', type=str, default='./data/audio_cremad_preprocessed',
                        help='Directorio de salida para archivos procesados')
    parser.add_argument('--audio_format', type=str, default='wav', choices=['wav', 'mp3'],
                        help='Formato de audio a procesar')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CREMA-D Audio Preprocessor")
    print("="*60)
    print(f"Input:  {args.cremad_root}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.audio_format}")
    print("="*60 + "\n")
    
    # Crear preprocesador
    preprocessor = CREMADAudioPreprocessor(
        cremad_root=args.cremad_root,
        output_dir=args.output_dir,
        audio_format=args.audio_format
    )
    
    # Procesar dataset
    preprocessor.process_dataset()
    
    print("\n" + "="*60)
    print("¡Listo! Ahora puedes usar AudioDataset con:")
    print(f"   AudioDataset('{args.output_dir}', include_sessions=['Session1'])")
    print("="*60)


if __name__ == "__main__":
    main()