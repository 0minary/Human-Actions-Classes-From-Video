"""
Data Loader pour la détection d'actions humaines dans des vidéos.

Utilise les modèles pré-entraînés sur Kinetics-400 de TorchVision
sans avoir besoin de télécharger le dataset complet (11.3 GB).
"""

import torch
import torchvision.models.video as models
from torchvision.models.video import (
    R3D_18_Weights,
    R3D_34_Weights,
    R3D_50_Weights,
    R2Plus1D_18_Weights,
    R2Plus1D_34_Weights,
    R2Plus1D_50_Weights,
    MC3_18_Weights,
    MC3_34_Weights,
    MC3_50_Weights,
    MViT_V1_B_Weights,
    MViT_V2_S_Weights,
    MViT_V2_B_Weights,
)
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class PretrainedKineticsModel:
    """
    Wrapper pour utiliser les modèles pré-entraînés sur Kinetics-400.
    
    Ces modèles ont été entraînés sur 400 classes d'actions et peuvent être utilisés
    pour :
    - Classification directe (400 classes Kinetics)
    - Feature extraction
    - Fine-tuning sur d'autres datasets (UCF-101, KTH, etc.)
    """
    
    # Mapping des modèles disponibles
    MODEL_MAP = {
        'r3d_18': (models.r3d_18, R3D_18_Weights.KINETICS400_V1),
        'r3d_34': (models.r3d_34, R3D_34_Weights.KINETICS400_V1),
        'r3d_50': (models.r3d_50, R3D_50_Weights.KINETICS400_V1),
        'r2plus1d_18': (models.r2plus1d_18, R2Plus1D_18_Weights.KINETICS400_V1),
        'r2plus1d_34': (models.r2plus1d_34, R2Plus1D_34_Weights.KINETICS400_V1),
        'r2plus1d_50': (models.r2plus1d_50, R2Plus1D_50_Weights.KINETICS400_V1),
        'mc3_18': (models.mc3_18, MC3_18_Weights.KINETICS400_V1),
        'mc3_34': (models.mc3_34, MC3_34_Weights.KINETICS400_V1),
        'mc3_50': (models.mc3_50, MC3_50_Weights.KINETICS400_V1),
        'mvit_v1_b': (models.mvit_v1_b, MViT_V1_B_Weights.KINETICS400_V1),
        'mvit_v2_s': (models.mvit_v2_s, MViT_V2_S_Weights.KINETICS400_V1),
        'mvit_v2_b': (models.mvit_v2_b, MViT_V2_B_Weights.KINETICS400_V1),
    }
    
    def __init__(self, model_name='r2plus1d_18', device=None):
        """
        Initialise le modèle pré-entraîné.
        
        Args:
            model_name: Nom du modèle ('r2plus1d_18', 'r3d_18', 'mvit_v1_b', etc.)
            device: Device ('cpu', 'cuda', ou None pour auto-détection)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model_name = model_name
        self.model = self._load_model()
        
    def _load_model(self):
        """Charge le modèle pré-entraîné sur Kinetics-400."""
        if self.model_name not in self.MODEL_MAP:
            available = ', '.join(self.MODEL_MAP.keys())
            raise ValueError(
                f"Modèle '{self.model_name}' non supporté. "
                f"Modèles disponibles: {available}"
            )
        
        model_fn, weights = self.MODEL_MAP[self.model_name]
        model = model_fn(weights=weights)
        model = model.to(self.device)
        model.eval()  # Mode évaluation par défaut
        
        # Afficher les informations
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Modèle {self.model_name} chargé avec succès!")
        print(f"  - Poids: Kinetics-400 (pré-entraîné)")
        print(f"  - Paramètres: {num_params:,}")
        print(f"  - Device: {self.device}")
        
        return model
    
    def prepare_video(self, frames, num_frames=16, size=(112, 112)):
        """
        Prépare un tenseur vidéo pour le modèle.
        
        Args:
            frames: Liste de frames (PIL Images, numpy arrays, ou chemins de fichiers)
            num_frames: Nombre de frames à utiliser (16 par défaut)
            size: Taille (H, W) pour le redimensionnement
        
        Returns:
            Tensor de forme (1, 3, T, H, W) prêt pour le modèle
        """
        # Normalisation pour Kinetics-400
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989]
            )
        ])
        
        processed = []
        for frame in frames[:num_frames]:
            # Gérer différents types d'input
            if isinstance(frame, str):
                frame = Image.open(frame).convert('RGB')
            elif not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame).convert('RGB')
            
            processed.append(transform(frame))
        
        # Si moins de frames que demandé, répéter la dernière
        while len(processed) < num_frames:
            processed.append(processed[-1])
        
        # Stack en tenseur: (T, C, H, W) -> (C, T, H, W) -> (1, C, T, H, W)
        video_tensor = torch.stack(processed, dim=0)  # (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # (1, C, T, H, W)
        
        return video_tensor.to(self.device)
    
    def predict(self, video_tensor, top_k=5):
        """
        Prédit les classes pour une vidéo.
        
        Args:
            video_tensor: Tensor (1, 3, T, H, W)
            top_k: Nombre de prédictions à retourner
        
        Returns:
            Liste de dictionnaires avec 'class_id' et 'probability'
        """
        with torch.no_grad():
            outputs = self.model(video_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=1)
            
            results = []
            for i in range(top_k):
                results.append({
                    'class_id': top_indices[0][i].item(),
                    'probability': top_probs[0][i].item()
                })
            return results
    
    def extract_features(self, video_tensor):
        """
        Extrait les features d'une vidéo (sans la dernière couche de classification).
        
        Args:
            video_tensor: Tensor (1, 3, T, H, W)
        
        Returns:
            Features vector (feature_dim,)
        """
        # Créer un modèle sans la dernière couche
        feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        feature_extractor.eval()
        
        with torch.no_grad():
            features = feature_extractor(video_tensor)
            features = features.squeeze()
            
            # Global average pooling si nécessaire
            if features.dim() > 1:
                # Average pooling spatio-temporel
                features = features.mean(dim=tuple(range(1, features.dim())))
        
        return features
    
    def prepare_for_finetuning(self, num_classes, freeze_backbone=False):
        """
        Prépare le modèle pour le fine-tuning sur un nouveau dataset.
        
        Args:
            num_classes: Nombre de classes du nouveau dataset (ex: 101 pour UCF-101)
            freeze_backbone: Si True, gèle les couches de base (feature extraction)
        
        Returns:
            Modèle prêt pour le fine-tuning
        """
        # Geler les couches de base si demandé
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"✓ Couches de base gelées (feature extraction mode)")
        else:
            print(f"✓ Toutes les couches entraînables (fine-tuning mode)")
        
        # Remplacer la dernière couche selon l'architecture
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(in_features, num_classes)
            self.model.fc.requires_grad = True
            print(f"✓ Couche de classification remplacée: {in_features} -> {num_classes}")
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = torch.nn.Linear(in_features, num_classes)
            self.model.head.requires_grad = True
            print(f"✓ Couche de classification remplacée: {in_features} -> {num_classes}")
        else:
            raise ValueError("Impossible de trouver la couche de classification à remplacer")
        
        self.model.train()  # Passer en mode entraînement
        return self.model
    
    def get_model_info(self):
        """Retourne des informations sur le modèle."""
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'device': self.device,
            'total_parameters': num_params,
            'trainable_parameters': trainable,
            'mode': 'training' if self.model.training else 'evaluation'
        }


# Exemple d'utilisation
if __name__ == "__main__":
    print("=" * 60)
    print("Chargement du modèle pré-entraîné sur Kinetics-400")
    print("=" * 60)
    
    # 1. Charger le modèle pré-entraîné (recommandé: r2plus1d_18)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kinetics_model = PretrainedKineticsModel(
        model_name='r2plus1d_18',
        device=device
    )
    
    print("\n" + "=" * 60)
    print("Informations du modèle:")
    print("=" * 60)
    info = kinetics_model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Modèle prêt à utiliser!")
    print("=" * 60)
    print("\nExemples d'utilisation:")
    print("  1. Classification: predictions = model.predict(video_tensor)")
    print("  2. Feature extraction: features = model.extract_features(video_tensor)")
    print("  3. Fine-tuning: model.prepare_for_finetuning(num_classes=101)")
