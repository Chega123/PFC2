import json
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from PIL import Image

class IEMOCAPDataset(Dataset):
    def __init__(self, json_file, processor, max_frames=8):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.processor = processor
        self.max_frames = max_frames
        
        # Validar videos
        valid_samples = []
        
        for sample in self.data:
            video_path = sample['video']
            if not os.path.exists(video_path):
                continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                continue
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if frame_count == 0:
                continue
            
            valid_samples.append(sample)
        
        self.data = valid_samples
        print(f"Dataset válido: {len(self.data)} videos\n")
        
    def extract_frames_as_pil(self, video_path):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se puede abrir: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            raise ValueError(f"Sin frames: {video_path}")
        
        # Índices uniformemente espaciados
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
            
            # BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Verificar no todo negro
            if frame_rgb.mean() < 1.0:
                continue
            
            # Convertir a PIL
            pil_image = Image.fromarray(frame_rgb)
            pil_frames.append(pil_image)
        
        cap.release()
        
        if len(pil_frames) == 0:
            raise ValueError(f"No frames válidos: {video_path}")
        
        return pil_frames
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        max_retries = 5
        current_idx = idx
        
        for attempt in range(max_retries):
            try:
                video_path = sample['video']
                
                # Extraer frames como PIL
                frames = self.extract_frames_as_pil(video_path)
                
                # Preparar conversación
                conversation = sample['conversations']
                user_content = conversation[0]['value'].replace('<video>\n', '')
                assistant_content = conversation[1]['value']
                
                # Usar múltiples imágenes como "video" para Qwen
                image_content = [{"type": "image", "image": frame} for frame in frames]
                
                messages = [
                    {
                        "role": "user",
                        "content": image_content + [{"type": "text", "text": user_content}]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_content}]
                    }
                ]
                
                # Aplicar template
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # Procesar con el processor
                inputs = self.processor(
                    text=[text],
                    images=frames,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Extraer tensores
                input_ids = inputs['input_ids'][0]
                attention_mask = inputs['attention_mask'][0]
                pixel_values = inputs.get('pixel_values', None)
                image_grid_thw = inputs.get('image_grid_thw', None)
                
                # Validar pixel_values
                if pixel_values is None:
                    raise ValueError(f"pixel_values es None")
                
                if isinstance(pixel_values, list):
                    if len(pixel_values) > 0:
                        pixel_values = pixel_values[0]
                    else:
                        raise ValueError(f"pixel_values vacío")
                
                # Validar que tengan valores razonables
                if isinstance(pixel_values, torch.Tensor):
                    mean_val = pixel_values.float().mean().item()
                    std_val = pixel_values.float().std().item()
                    if abs(mean_val) < 1e-6 and std_val < 1e-6:
                        raise ValueError(f"pixel_values inválidos (mean={mean_val}, std={std_val})")
                
                # Validar y procesar image_grid_thw
                if image_grid_thw is None:
                    raise ValueError(f"image_grid_thw es None")
                
                if isinstance(image_grid_thw, list):
                    if len(image_grid_thw) > 0:
                        image_grid_thw = image_grid_thw[0]
                    else:
                        raise ValueError(f"image_grid_thw vacío")
                
                # Asegurar que tiene el shape correcto [num_images, 3]
                if image_grid_thw.dim() == 1:
                    # Si es [3] -> reshape a [1, 3]
                    image_grid_thw = image_grid_thw.unsqueeze(0)
                elif image_grid_thw.dim() == 3:
                    # Si es [1, num_images, 3] -> squeeze primera dim
                    image_grid_thw = image_grid_thw.squeeze(0)
                
                # Crear labels
                labels = input_ids.clone()
                
                # Encontrar inicio de respuesta del asistente
                user_messages = [
                    {
                        "role": "user",
                        "content": image_content + [{"type": "text", "text": user_content}]
                    }
                ]
                
                user_text = self.processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                user_inputs = self.processor(
                    text=[user_text],
                    images=frames,
                    padding=True,
                    return_tensors="pt"
                )
                
                user_length = user_inputs['input_ids'].shape[1]
                
                # Enmascarar prompt del usuario
                labels[:user_length] = -100
                
                # Verificar tokens válidos
                valid_tokens = (labels != -100).sum().item()
                if valid_tokens == 0:
                    labels[-10:] = input_ids[-10:]
                
                result = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'pixel_values': pixel_values,
                    'image_grid_thw': image_grid_thw,  # IMPORTANTE: siempre incluir
                    'labels': labels
                }
                
                return result
                
            except Exception as e:
                print(f"ERROR sample {current_idx} (intento {attempt+1}/{max_retries}): {e}")
                current_idx = (current_idx + 1) % len(self)
                sample = self.data[current_idx]
                
                if attempt == max_retries - 1:
                    raise e


def collate_fn(batch):

    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Max length para padding
    max_length = max([item['input_ids'].shape[0] for item in batch])
    
    input_ids = []
    attention_masks = []
    pixel_values = []
    image_grid_thws = []  # Lista, NO tensor
    labels = []
    
    for item in batch:
        # Pad secuencias
        pad_length = max_length - item['input_ids'].shape[0]
        
        input_ids.append(torch.cat([
            item['input_ids'],
            torch.zeros(pad_length, dtype=torch.long)
        ]))
        
        attention_masks.append(torch.cat([
            item['attention_mask'],
            torch.zeros(pad_length, dtype=torch.long)
        ]))
        
        labels.append(torch.cat([
            item['labels'],
            torch.full((pad_length,), -100, dtype=torch.long)
        ]))
        
        # Pixel values
        if item['pixel_values'] is not None:
            pixel_values.append(item['pixel_values'])
        
        # Image grid thw - MANTENER COMO LISTA
        if item['image_grid_thw'] is not None:
            image_grid_thws.append(item['image_grid_thw'])
    
    if len(pixel_values) == 0:
        print("ERROR: Batch sin pixel_values")
        return None
    
    if len(image_grid_thws) == 0:
        print("ERROR: Batch sin image_grid_thw")
        return None
    
    # Stack pixel_values (todos deben tener mismo shape)
    try:
        pixel_values_tensor = torch.cat(pixel_values, dim=0)
    except Exception as e:
        print(f"WARNING: No se pudo concatenar pixel_values, usando lista: {e}")
        pixel_values_tensor = pixel_values

    # CRÍTICO: Concatenar todos los image_grid_thw en un solo tensor
    # El modelo espera [total_images, 3] donde total_images = sum de todos los frames
    try:
        image_grid_thw_tensor = torch.cat(image_grid_thws, dim=0)
        #print(f"  Final image_grid_thw: {image_grid_thw_tensor.shape}")
    except Exception as e:
        print(f"ERROR concatenando image_grid_thw: {e}")
        image_grid_thw_tensor = image_grid_thws[0]
    
    result = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'pixel_values': pixel_values_tensor,
        'image_grid_thw': image_grid_thw_tensor,  # Tensor concatenado
        'labels': torch.stack(labels)
    }
    
    return result