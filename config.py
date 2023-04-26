import albumentations as A
import torch

from albumentations.pytorch import ToTensorV2


CHECKPOINT_FILE = './checkpoints/x_ray_model.pth.tar'
DATASET_PATH = './dataset'
IMAGES_DATASET = './dataset/images'
TRANSFORMER_MODEL = 'bert-base-uncased'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
PIN_MEMORY = False
VOCAB_THRESHOLD = 2

GRAD_CLIP = True

FEATURES_SIZE = 1024
EMBED_SIZE = 768
HIDDEN_SIZE = 512

LEARNING_RATE = 4e-5
EPOCHS = 10

LOAD_MODEL = True
SAVE_MODEL = True

basic_transforms = A.Compose([
    A.Resize(
        height=256,
        width=256
    ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2()
])
