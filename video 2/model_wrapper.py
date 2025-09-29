""" 
# model_wrapper.py - Versión compatible con fusión
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image


class VideoQwenWrapper(nn.Module):
    def __init__(self,
                 model_name: str,
                 device="cuda",
                 num_virtual_tokens=20,
                 num_classes=4,
                 init_prompt="The emotion in the video is:"):
        super().__init__()

        self.device = device
        self.init_prompt = init_prompt
        self.num_classes = num_classes

        # Configuración de cuantización
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Cargar modelo y processor
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            load_in_4bit=True,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # PEFT con LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

        # Cabeza de clasificación
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, frames, labels=None):
        batch_size = len(frames)

        # Procesar cada video en el batch
        all_texts = []
        all_images = []
        
        for i in range(batch_size):
            video_frames = []
            for img in frames[i]:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img.astype("uint8"))
                video_frames.append(img)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in video_frames],
                        {"type": "text", "text": self.init_prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_texts.append(text)
            all_images.append(video_frames)
        
        # Procesar inputs
        inputs = self.processor(
            text=all_texts,
            images=all_images,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Forward
        outputs = self.model(**inputs, output_hidden_states=True)

        # Extraer último hidden state
        last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden]
        cls_vec = last_hidden[:, -1, :]          # [B, hidden]
        logits = self.classifier(cls_vec)        # [B, num_classes]

        # Calcular loss si hay labels
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

    def extract_features(self, frames):

        batch_size = len(frames)

        # Procesar cada video en el batch
        all_texts = []
        all_images = []
        
        for i in range(batch_size):
            video_frames = []
            for img in frames[i]:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img.astype("uint8"))
                video_frames.append(img)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in video_frames],
                        {"type": "text", "text": self.init_prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_texts.append(text)
            all_images.append(video_frames)
        
        # Procesar inputs
        inputs = self.processor(
            text=all_texts,
            images=all_images,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Forward sin gradientes para extracción de features
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden]
            features = last_hidden[:, -1, :]          # [B, hidden_size]
            
            # Si hidden_size != 768, proyectar a 768 para compatibilidad
            if features.size(-1) != 768:
                if not hasattr(self, 'feature_projector'):
                    self.feature_projector = nn.Linear(features.size(-1), 768).to(self.device)
                features = self.feature_projector(features)
            
            return features  # [batch_size, 768]


# Clase compatible con tu código de fusión existente
class VideoEmbeddingExtractor(VideoQwenWrapper):

    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", hidden_size=768, 
                 num_layers=1, dropout=0.0, num_frozen_layers=0, device="cuda"):
        super().__init__(
            model_name=model_name,
            device=device,
            num_classes=4
        )
        # Estos parámetros se ignoran pero se mantienen para compatibilidad
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_frozen_layers = num_frozen_layers """

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image


class VideoQwenWrapper(nn.Module):
    def __init__(self,
                 model_name: str,
                 device="cuda",
                 num_classes=4,
                 init_prompt="The emotion in the video is:"):
        super().__init__()

        self.device = device
        self.init_prompt = init_prompt
        self.num_classes = num_classes

        # Configuración de cuantización
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Cargar modelo y processor
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            load_in_4bit=True,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # PEFT con LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

        # Cabeza de clasificación
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, frames, labels=None):
        """Forward para entrenamiento - devuelve loss y logits"""
        batch_size = len(frames)

        # Procesar cada video en el batch
        all_texts = []
        all_images = []
        
        for i in range(batch_size):
            video_frames = []
            for img in frames[i]:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img.astype("uint8"))
                video_frames.append(img)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in video_frames],
                        {"type": "text", "text": self.init_prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_texts.append(text)
            all_images.append(video_frames)
        
        # Procesar inputs
        inputs = self.processor(
            text=all_texts,
            images=all_images,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Forward
        outputs = self.model(**inputs, output_hidden_states=True)

        # Extraer último hidden state
        last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden]
        cls_vec = last_hidden[:, -1, :]          # [B, hidden]
        logits = self.classifier(cls_vec)        # [B, num_classes]

        # Calcular loss si hay labels
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

    def extract_features(self, frames):
        """
        NUEVO MÉTODO para compatibilidad con fusión
        Devuelve embeddings de 768 dimensiones para usar en HierarchicalFusionModule
        """
        batch_size = len(frames)

        # Procesar cada video en el batch
        all_texts = []
        all_images = []
        
        for i in range(batch_size):
            video_frames = []
            for img in frames[i]:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img.astype("uint8"))
                video_frames.append(img)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in video_frames],
                        {"type": "text", "text": self.init_prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_texts.append(text)
            all_images.append(video_frames)
        
        # Procesar inputs
        inputs = self.processor(
            text=all_texts,
            images=all_images,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Forward sin gradientes para extracción de features
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden]
            features = last_hidden[:, -1, :]          # [B, hidden_size]
            
            # Si hidden_size != 768, proyectar a 768 para compatibilidad
            if features.size(-1) != 768:
                if not hasattr(self, 'feature_projector'):
                    self.feature_projector = nn.Linear(features.size(-1), 768).to(self.device)
                features = self.feature_projector(features)
            
            return features  # [batch_size, 768]


# Clase compatible con tu código de fusión existente
class VideoEmbeddingExtractor(VideoQwenWrapper):
    """
    Wrapper para mantener compatibilidad total con tu código de fusión existente
    """
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", hidden_size=768, 
                 num_layers=1, dropout=0.0, num_frozen_layers=0, device="cuda"):
        super().__init__(
            model_name=model_name,
            device=device,
            num_classes=4
        )
        # Estos parámetros se ignoran pero se mantienen para compatibilidad
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_frozen_layers = num_frozen_layers
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device="cuda"):
        """
        Cargar modelo desde checkpoint .pth para compatibilidad con fusión
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extraer configuración del modelo
        model_config = checkpoint.get('model_config', {})
        model_name = model_config.get('model_name', 'Qwen/Qwen2-VL-2B-Instruct')
        
        # Crear instancia
        model = cls(
            model_name=model_name,
            device=device,
            hidden_size=768  # Fijo para compatibilidad con fusión
        )
        
        # Cargar parámetros entrenados
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        return model