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


class QwenBatchEmbeddingExtractor:

    def __init__(self, base_model_name, adapter_path, device='cuda', max_frames=3, batch_size=4):
        self.device = device
        self.max_frames = max_frames
        self.batch_size = batch_size
        
        print(f"\nCargando modelo base: {base_model_name}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print(f"Cargando adaptador LoRA: {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for module in self.model.modules():
            if hasattr(module, 'dropout'):
                module.dropout = 0.0
        
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
            raise ValueError(f"No frames validos: {video_path}")
        
        return pil_frames
    
    def prepare_batch_inputs(self, samples_batch):
        batch_frames = []
        batch_texts = []
        valid_indices = []
        
        for idx, sample in enumerate(samples_batch):
            try:
                video_path = sample['video']
                frames = self.extract_frames(video_path)
                
                conversation = sample['conversations']
                
                system_content = None
                user_content = None
                
                for msg in conversation:
                    if msg['from'] == 'system':
                        system_content = msg['value']
                    elif msg['from'] == 'user':
                        user_content = msg['value'].replace('<video>\n', '')
                
                image_content = [{"type": "image", "image": frame} for frame in frames]
                
                messages = []
                
                if system_content:
                    messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": system_content}]
                    })
                
                messages.append({
                    "role": "user",
                    "content": image_content + [{"type": "text", "text": user_content}]
                })
                
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                batch_frames.append(frames)
                batch_texts.append(text)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"Error preparando sample {idx}: {e}")
                continue
        
        if len(batch_frames) == 0:
            return None, []
        
        return (batch_frames, batch_texts), valid_indices
    
    @torch.no_grad()
    def extract_embeddings_batch(self, samples_batch, pooling='last'):
        batch_data, valid_indices = self.prepare_batch_inputs(samples_batch)
        
        if batch_data is None:
            return [None] * len(samples_batch)
        
        batch_frames, batch_texts = batch_data
        embeddings_list = [None] * len(samples_batch)
        
        for i, (frames, text) in enumerate(zip(batch_frames, batch_texts)):
            try:
                inputs = self.processor(
                    text=[text],
                    images=frames,
                    padding=True,
                    return_tensors="pt"
                )
                
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                pixel_values = inputs['pixel_values']
                
                if isinstance(pixel_values, list):
                    pixel_values = pixel_values[0]
                pixel_values = pixel_values.to(self.device)
                
                image_grid_thw = inputs['image_grid_thw']
                if isinstance(image_grid_thw, list):
                    image_grid_thw = image_grid_thw[0]
                if image_grid_thw.dim() == 1:
                    image_grid_thw = image_grid_thw.unsqueeze(0)
                elif image_grid_thw.dim() == 3:
                    image_grid_thw = image_grid_thw.squeeze(0)
                image_grid_thw = image_grid_thw.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True
                )
                
                last_hidden = outputs.hidden_states[-1].float()
                
                if pooling == 'last':
                    embedding = last_hidden[0, -1, :].cpu()
                elif pooling == 'mean':
                    embedding = last_hidden[0].mean(dim=0).cpu()
                elif pooling == 'max':
                    embedding = last_hidden[0].max(dim=0)[0].cpu()
                else:
                    raise ValueError(f"Pooling '{pooling}' no soportado")
                
                hidden_size = embedding.shape[0]
                if hidden_size != 768:
                    if not hasattr(self, 'projector'):
                        print(f"\n[INFO] Creando proyector: {hidden_size} -> 768")
                        self.projector = torch.nn.Linear(hidden_size, 768).to(self.device)
                        torch.nn.init.xavier_uniform_(self.projector.weight)
                    
                    embedding = self.projector(embedding.to(self.device)).cpu()
                
                embeddings_list[valid_indices[i]] = embedding
                
                del input_ids, attention_mask, pixel_values, image_grid_thw
                del outputs, last_hidden
                
            except Exception as e:
                print(f"Error procesando video {i}: {e}")
                continue
        
        return embeddings_list


def detect_emotion_classes(json_path):
    """Detecta automaticamente las clases de emociones del JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    emotions = set()
    for sample in data:
        for msg in sample['conversations']:
            if msg['from'] == 'assistant':
                emotions.add(msg['value'].strip().lower())
    
    emotions = sorted(emotions)
    emotion_map = {emotion: i for i, emotion in enumerate(emotions)}
    
    print(f"Emociones detectadas: {emotions}")
    print(f"Mapeo: {emotion_map}\n")
    
    return emotion_map


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
    print(f"EXTRACCION BATCH - QWEN2.5-VL")
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
    
    for json_path in json_paths:
        print(f"\n{'='*70}")
        print(f"Procesando: {json_path}")
        print(f"{'='*70}")
        
        emotion_map = detect_emotion_classes(json_path)
        
        split_name = os.path.splitext(os.path.basename(json_path))[0]
        split_output_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"Total samples: {len(data)}\n")
        
        successful = 0
        failed = []
        skipped = 0
        
        with tqdm(total=len(data), desc=f"Extrayendo {split_name}", unit="video") as pbar:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                to_process = []
                for item in batch:
                    save_path = os.path.join(split_output_dir, f"{item['id']}.pt")
                    if os.path.exists(save_path):
                        skipped += 1
                        pbar.update(1)
                    else:
                        to_process.append(item)
                
                if len(to_process) == 0:
                    continue
                
                embeddings = extractor.extract_embeddings_batch(to_process, pooling=pooling)
                
                for item, embedding in zip(to_process, embeddings):
                    video_id = item['id']
                    
                    emotion = None
                    for msg in item['conversations']:
                        if msg['from'] == 'assistant':
                            emotion = msg['value'].strip().lower()
                            break
                    
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
                        reason = "Embedding failed" if embedding is None else f"Unknown emotion: {emotion}"
                        failed.append((video_id, reason))
                    
                    pbar.update(1)
                
                if i % (batch_size * 5) == 0 and i > 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
                pbar.set_postfix({
                    'exitosos': successful,
                    'fallidos': len(failed),
                    'saltados': skipped
                })
        
        print(f"\n{split_name} completado:")
        print(f"  Exitosos: {successful}/{len(data)}")
        print(f"  Fallidos: {len(failed)}")
        print(f"  Saltados (ya existen): {skipped}")
        
        if failed:
            print(f"\nVideos fallidos:")
            for vid, reason in failed[:10]:
                print(f"  - {vid}: {reason}")
            if len(failed) > 10:
                print(f"  ... y {len(failed) - 10} mas")
    
    print(f"\n{'='*70}")
    print(f"EXTRACCION COMPLETADA!")
    print(f"Ubicacion: {output_dir}")
    print(f"{'='*70}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CONFIGURACION - Cambia aqui para IEMOCAP o CREMA-D
    DATASET = "cremad"  # Opciones: "iemocap" o "cremad"
    
    if DATASET == "iemocap":
        json_paths = ["train.json", "val.json"]
        checkpoint = "./qwen_iemocap_checkpoints/final_model"
        output_dir = "./data/video_embeddings_iemocap"
    elif DATASET == "cremad":
        json_paths = ["train.json", "val.json", "test.json"]
        checkpoint = "./qwen_iemocap_checkpoints/final_model"
        output_dir = "./data/video_embeddings_cremad"
    else:
        print(f"ERROR: Dataset '{DATASET}' no reconocido")
        return
    
    batch_size = 16
    max_frames = 3
    pooling = 'last'
    
    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"ERROR: {json_path} no existe")
            return
    
    if not os.path.exists(checkpoint):
        print(f"ERROR: {checkpoint} no existe")
        return
    
    extract_from_json_with_batching(
        json_paths=json_paths,
        qwen_checkpoint_path=checkpoint,
        output_dir=output_dir,
        max_frames=max_frames,
        batch_size=batch_size,
        pooling=pooling,
        device=device
    )
    
    print("\nListo! Embeddings guardados.")
    print(f"\nPara usar en fusion:")
    print(f"  data = torch.load('{output_dir}/train/sample_id.pt')")
    print(f"  embedding = data['embedding']  # shape: [768]")
    print(f"  label = data['label']  # ID numerico")
    print(f"  emotion = data['emotion']  # nombre de emocion")


if __name__ == "__main__":
    main()