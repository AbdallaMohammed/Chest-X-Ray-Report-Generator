import os
import torch
import config
import utils
import numpy as np

from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split as sklearn_train_test_split


def text_preprocessing(text):
    return text.lower()


def train_test_split(dataset, test_size=0.25, random_state=44):
    train_idx, test_idx = sklearn_train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        random_state=random_state
    )

    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def save_checkpoint(checkpoint):
    print('=> Saving checkpoint')

    torch.save(checkpoint, config.CHECKPOINT_FILE)


def load_checkpoint(model, optimizer=None):
    print('=> Loading checkpoint')

    checkpoint = torch.load(config.CHECKPOINT_FILE)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']


def can_load_checkpoint():
    return os.path.exists(config.CHECKPOINT_FILE) and config.LOAD_MODEL


def check_accuracy(dataset, vocab, model, epoch):
    print('=> Testing')

    model.eval()

    bleu1_score = []
    bleu2_score = []
    bleu3_score = []
    bleu4_score = []

    for item_id, (image, caption) in enumerate(tqdm(dataset)):
        image = image.to(config.DEVICE)

        caption = utils.text_preprocessing(' '.join([vocab.itos[idx] for idx in caption.numpy()]))
        generated = utils.text_preprocessing(' '.join(model.generate_caption(image.unsqueeze(0), vocab, max_length=150)))

        bleu1_score.append(
            sentence_bleu(
                [caption.split()],
                generated.split(),
                smoothing_function=SmoothingFunction().method4,
                weights=(1.0, 0, 0, 0)
            )
        )

        bleu2_score.append(
            sentence_bleu(
                [caption.split()],
                generated.split(),
                smoothing_function=SmoothingFunction().method4,
                weights=(0.5, 0.5, 0, 0)
            )
        )

        bleu3_score.append(
            sentence_bleu(
                [caption.split()],
                generated.split(),
                smoothing_function=SmoothingFunction().method4,
                weights=(0.33, 0.33, 0.33, 0)
            )
        )

        bleu4_score.append(
            sentence_bleu(
                [caption.split()],
                generated.split(),
                smoothing_function=SmoothingFunction().method4,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
        )

    print(f'Epoch[{epoch}] => BLEU 1: {np.mean(bleu1_score)}')
    print(f'Epoch[{epoch}] => BLEU 2: {np.mean(bleu2_score)}')
    print(f'Epoch[{epoch}] => BLEU 3: {np.mean(bleu3_score)}')
    print(f'Epoch[{epoch}] => BLEU 4: {np.mean(bleu4_score)}')
