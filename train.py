import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config
import utils

from tqdm import tqdm
from torch.utils.data import DataLoader
from model import EncoderDecoderNet
from dataset import XRayDataset, CollateDataset


def train_epoch(loader, model, optimizer, loss_fn, epoch):
    model.train()

    losses = []

    loader = tqdm(loader)

    for img, captions in loader:
        img = img.to(config.DEVICE)
        captions = captions.to(config.DEVICE)

        output = model(img, captions[:-1])

        loss = loss_fn(
            output.reshape(-1, output.shape[2]),
            captions.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loader.set_postfix(loss=loss.item())

        losses.append(loss.item())

    if config.SAVE_MODEL:
        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        })

    print(f'Epoch[{epoch}]: Loss {np.mean(losses)}')


def main():
    all_dataset = XRayDataset(
        root=config.DATASET_PATH,
        transform=config.basic_transforms,
        freq_threshold=config.VOCAB_THRESHOLD,
        text_preprocessing=utils.text_preprocessing
    )

    train_dataset, test_dataset = utils.train_test_split(dataset=all_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        shuffle=True,
        collate_fn=CollateDataset(pad_idx=all_dataset.vocab.stoi['<PAD>']),
    )

    model = EncoderDecoderNet(
        embed_size=512,
        hidden_size=512,
        vocab_size=len(all_dataset.vocab)
    )
    model = model.to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=all_dataset.vocab.stoi['<PAD>'])

    starting_epoch = 1

    if utils.can_load_checkpoint():
        starting_epoch = utils.load_checkpoint(model, optimizer)

    for epoch in range(starting_epoch, config.EPOCHS):
        train_epoch(
            train_loader,
            model,
            optimizer,
            loss_fn,
            epoch
        )

        utils.check_accuracy(
            test_dataset,
            all_dataset.vocab,
            model,
            epoch
        )


if __name__ == '__main__':
    main()