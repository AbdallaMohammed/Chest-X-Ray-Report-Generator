import albumentations as A
import torch

from albumentations.pytorch import ToTensorV2


CHECKPOINT_FILE = './checkpoints/x_ray_model.pth.tar'
DATASET_PATH = './dataset'
IMAGES_DATASET = './dataset/images'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
PIN_MEMORY = False
VOCAB_THRESHOLD = 3
EMBED_SIZE = 512
HIDDEN_SIZE = 512
LEARNING_RATE = 5e-4
EPOCHS = 100

LOAD_MODEL = True
SAVE_MODEL = True

basic_transforms = A.Compose([
    A.Resize(
        height=224,
        width=224
    ),
    A.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    ToTensorV2()
])
