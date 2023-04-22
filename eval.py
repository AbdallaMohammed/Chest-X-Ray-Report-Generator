import config
import utils
import numpy as np

from tqdm import tqdm
from dataset import XRayDataset
from model import EncoderDecoderNet
from nltk.translate.bleu_score import sentence_bleu


def check_accuracy(dataset, vocab, model):
    print('=> Testing')

    model.eval()

    bleu1_score = []
    bleu2_score = []
    bleu3_score = []
    bleu4_score = []

    for _, (image, caption) in enumerate(tqdm(dataset)):
        image = image.to(config.DEVICE)

        generated = model.generate_caption(image.unsqueeze(0), vocab, max_length=len(caption) + 1)

        bleu1_score.append(
            sentence_bleu([caption.split()], generated, weights=(1, 0, 0, 0))
        )

        bleu2_score.append(
            sentence_bleu([caption.split()], generated, weights=(0.5, 0.5, 0, 0))
        )

        bleu3_score.append(
            sentence_bleu([caption.split()], generated, weights=(0.33, 0.33, 0.33, 0))
        )

        bleu4_score.append(
            sentence_bleu([caption.split()], generated, weights=(0.25, 0.25, 0.25, 0.25))
        )

    print(f'=> BLEU 1: {np.mean(bleu1_score)}')
    print(f'=> BLEU 2: {np.mean(bleu2_score)}')
    print(f'=> BLEU 3: {np.mean(bleu3_score)}')
    print(f'=> BLEU 4: {np.mean(bleu4_score)}')


def main():
    all_dataset = XRayDataset(
        root=config.DATASET_PATH,
        transform=config.basic_transforms,
        freq_threshold=config.VOCAB_THRESHOLD,
        raw_caption=True
    )

    model = EncoderDecoderNet(
        features_size=config.FEATURES_SIZE,
        embed_size=config.EMBED_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        vocab_size=len(all_dataset.vocab)
    )
    model = model.to(config.DEVICE)

    _, test_dataset = utils.train_test_split(dataset=all_dataset)

    utils.load_checkpoint(model)

    check_accuracy(
        test_dataset,
        all_dataset.vocab,
        model
    )


if __name__ == '__main__':
    main()
