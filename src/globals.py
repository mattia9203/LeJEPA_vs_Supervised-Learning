"""Global constants and project-wide settings."""

# Dataset
NUM_CLASSES = 30  # Selected ImageNet-100 subset classes used by this project
IMAGE_SIZE = 224  # Pretrained ViTs expect 224x224 input
PATCH_SIZE = 16   # ViT-S/16 patch size -> 14x14 = 196 patch tokens

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model identifiers
SUPERVISED_VIT_MODEL = "vit_small_patch16_224"
LEJEPA_VIT_MODEL = "OK-AI/lejepa-vits16-pretrain-in1k"
LEJEPA_VIT_REVISION = "cc7022877d51494709ef398d437fb8619349e0f9"

# Supported model types
MODEL_TYPE_SUPERVISED = "supervised_vit"
MODEL_TYPE_LEJEPA = "lejepa_vit"
MODEL_TYPE_SUPERVISED_CNN = "supervised_cnn"
MODEL_TYPE_LEJEPA_CNN = "lejepa_cnn"
RESNET50_EMBED_DIM = 2048
