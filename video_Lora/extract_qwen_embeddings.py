"""
Extracción de embeddings de Qwen2.5-VL con procesamiento por BATCHES (más rápido)
"""
import torch
import os
import json
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import cv2
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor


class QwenBatchEmbeddingExtractor:

    def __init__(self, base_model_name, adapter_path, device='cuda', max_frames=3, batch_size=4):
        self.device = device
        self.max_frames = max_frames
        self.batch_size = batch_size
        
        print(f"\nCargando modelo base: {base_model_name}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print(f" Cargando adaptador LoRA: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"Cargando processor")
        self.processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        print(f"Modelo cargado | Batch size: {batch_size}\n")
    
    def extract_frames(self, video_path):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se puede abrir: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            raise ValueError(f"Sin frames: {video_path}")
        
        if frame_count <= self.max_frames:
            frame_indices = list(range(frame_count))
        else:
            frame_indices = np.linspace(0, frame_count - 1, self.max_frames, dtype=int)
        
        pil_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_rgb.mean() < 1.0:
                continue
            
            pil_frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        
        if len(pil_frames) == 0:
            raise ValueError(f"No frames válidos: {video_path}")
        
        return pil_frames
    
    def prepare_batch_inputs(self, video_paths_batch):

        batch_frames = []
        batch_texts = []
        valid_indices = []
        
        for idx, video_path in enumerate(video_paths_batch):
            try:
                frames = self.extract_frames(video_path)
                
                # Preparar mensaje
                image_content = [{"type": "image", "image": frame} for frame in frames]
                messages = [{
                    "role": "user",
                    "content": image_content + [{"type": "text", "text": "Describe the emotion."}]
                }]
                
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                batch_frames.append(frames)
                batch_texts.append(text)
                valid_indices.append(idx)
                
            except Exception as e:

                continue
        
        if len(batch_frames) == 0:
            return None, []

        
        return (batch_frames, batch_texts), valid_indices
    
    @torch.no_grad()
    def extract_embeddings_batch(self, video_paths_batch, pooling='last'):
        """
        Extrae embeddings de un batch de videos
        Retorna: lista de embeddings (None si falló)
        """
        batch_data, valid_indices = self.prepare_batch_inputs(video_paths_batch)
        
        if batch_data is None:
            return [None] * len(video_paths_batch)
        
        batch_frames, batch_texts = batch_data
        embeddings_list = [None] * len(video_paths_batch)
        
        # Procesar cada video del batch (Qwen no soporta batch real para videos mixtos)
        for i, (frames, text) in enumerate(zip(batch_frames, batch_texts)):
            try:
                # Procesar
                inputs = self.processor(
                    text=[text],
                    images=frames,
                    padding=True,
                    return_tensors="pt"
                )
                
                inputs = {
                    'input_ids': inputs['input_ids'].to(self.device),
                    'attention_mask': inputs['attention_mask'].to(self.device),
                    'pixel_values': inputs['pixel_values'].to(self.device),
                    'image_grid_thw': inputs['image_grid_thw'].to(self.device)
                }
                
                # Forward pass
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Extraer hidden state y convertir a float32
                last_hidden = outputs.hidden_states[-1].float()
                
                # Pooling
                if pooling == 'last':
                    embedding = last_hidden[0, -1, :].cpu()
                elif pooling == 'mean':
                    embedding = last_hidden[0].mean(dim=0).cpu()
                elif pooling == 'max':
                    embedding = last_hidden[0].max(dim=0)[0].cpu()
                else:
                    raise ValueError(f"Pooling '{pooling}' no soportado")
                
                # Proyectar si es necesario
                hidden_size = embedding.shape[0]
                if hidden_size != 768:
                    if not hasattr(self, 'projector'):
                        print(f"\n[INFO] Creando proyector: {hidden_size} -> 768")
                        self.projector = torch.nn.Linear(hidden_size, 768).to(self.device)
                        torch.nn.init.xavier_uniform_(self.projector.weight)
                    
                    embedding = self.projector(embedding.to(self.device)).cpu()
                
                # Guardar en la posición correcta
                embeddings_list[valid_indices[i]] = embedding
                
                # Limpiar
                del inputs, outputs, last_hidden
                
            except Exception as e:
                print(f"Error en video {i}: {e}")
                continue
        
        return embeddings_list


def extract_from_json_with_batching(
    json_paths,
    qwen_checkpoint_path,
    output_dir,
    base_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    max_frames=3,
    batch_size=4,
    pooling='last',
    device='cuda'
):

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"EXTRACCIÓN BATCH - QWEN2.5-VL")
    print(f"{'='*70}")
    print(f"Checkpoint: {qwen_checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Max frames: {max_frames}")
    print(f"Pooling: {pooling}")
    print(f"{'='*70}\n")
    
    extractor = QwenBatchEmbeddingExtractor(
        base_model_name=base_model_name,
        adapter_path=qwen_checkpoint_path,
        device=device,
        max_frames=max_frames,
        batch_size=batch_size
    )
    
    emotion_map = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "excited": 1}
    
    for json_path in json_paths:
        print(f"\n{'='*70}")
        print(f"Procesando: {json_path}")
        print(f"{'='*70}")
        
        split_name = os.path.splitext(os.path.basename(json_path))[0]
        split_output_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"Total samples: {len(data)}\n")
        
        successful = 0
        failed = []
        
        # Procesar en batches con barra de progreso
        with tqdm(total=len(data), desc=f"Extrayendo {split_name}", unit="video") as pbar:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # Verificar cuáles ya existen
                to_process = []
                for item in batch:
                    save_path = os.path.join(split_output_dir, f"{item['id']}.pt")
                    if os.path.exists(save_path):
                        successful += 1
                        pbar.update(1)
                    else:
                        to_process.append(item)
                
                if len(to_process) == 0:
                    continue
                
                # Extraer embeddings del batch
                batch_video_paths = [item['video'] for item in to_process]
                embeddings = extractor.extract_embeddings_batch(batch_video_paths, pooling=pooling)
                
                # Guardar resultados
                for item, embedding in zip(to_process, embeddings):
                    video_id = item['id']
                    emotion = item['conversations'][1]['value'].strip().lower()
                    label = emotion_map.get(emotion, -1)
                    
                    if embedding is not None and label != -1:
                        save_path = os.path.join(split_output_dir, f"{video_id}.pt")
                        torch.save({
                            'embedding': embedding,
                            'label': label,
                            'video_id': video_id,
                            'emotion': emotion,
                            'video_path': item['video']
                        }, save_path)
                        successful += 1
                    else:
                        failed.append((video_id, "Extraction failed"))
                    
                    pbar.update(1)
                
                # Limpiar memoria cada 20 videos
                if i % (batch_size * 5) == 0 and i > 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Actualizar descripción con stats
                pbar.set_postfix({'exitosos': successful, 'fallidos': len(failed)})
        
        print(f"\n{split_name} completado:")
        print(f"  ✓ Exitosos: {successful}/{len(data)}")
        print(f"  ✗ Fallidos: {len(failed)}")
        
        if failed:
            print(f"\nPrimeros 5 videos fallidos:")
            for vid, reason in failed[:5]:
                print(f"  - {vid}: {reason}")
    
    print(f"\n{'='*70}")
    print(f"¡EXTRACCIÓN COMPLETADA!")
    print(f"Ubicación: {output_dir}")
    print(f"{'='*70}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuración
    base_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_checkpoint_path = "./qwen_iemocap_checkpoints/final_model"
    output_dir = "./data/video_embeddings_qwen"
    
    json_paths = ["train.json", "val.json"]
    
    # Verificar que existen
    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"ERROR: {json_path} no existe")
            return
    
    if not os.path.exists(qwen_checkpoint_path):
        print(f"ERROR: {qwen_checkpoint_path} no existe")
        return
    
    # Extraer con batching
    extract_from_json_with_batching(
        json_paths=json_paths,
        qwen_checkpoint_path=qwen_checkpoint_path,
        output_dir=output_dir,
        base_model_name=base_model_name,
        max_frames=3,
        batch_size=16,  # Ajusta según tu VRAM (4-8 es razonable)
        pooling='last',
        device=device
    )
    
    print("\n✓ ¡Listo! Embeddings guardados.")
    print(f"\nPara usar en fusión:")
    print(f"  data = torch.load('data/video_embeddings_qwen/train/Ses01F_impro01_F000.pt')")
    print(f"  embedding = data['embedding']  # [768]")


if __name__ == "__main__":
    main()