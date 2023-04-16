import config
import utils

from dataset import XRayDataset
from model import EncoderDecoderNet
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu


def main():
    all_dataset = XRayDataset(
        root=config.DATASET_PATH,
        transform=config.basic_transforms,
        freq_threshold=config.VOCAB_THRESHOLD,
        text_preprocessing=utils.text_preprocessing,
        raw_caption=True
    )

    model = EncoderDecoderNet(
        embed_size=512,
        hidden_size=512,
        vocab_size=len(all_dataset.vocab)
    )
    model = model.to(config.DEVICE)

    utils.load_checkpoint(model)

    img = all_dataset[888][0].to(config.DEVICE)
    caption = all_dataset[888][1]

    generated_caption = ' '.join(model.generate_caption(img.unsqueeze(0), all_dataset.vocab, max_length=500))

    print(caption)
    print(generated_caption)


main()
