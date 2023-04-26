import config
import utils
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from dataset import CollateDataset


def train_epoch(loader, test_loader, model, optimizer, loss_fn, vocabulary, scheduler=None, epoch=1):
    model.train()

    losses = []

    loader = tqdm(loader)

    for img, captions, masks in loader:
        img = img.to(config.DEVICE)
        captions = captions.to(config.DEVICE)
        masks = masks.to(config.DEVICE)

        output = model(img, captions, masks)

        output = output.contiguous().view(-1, output.shape[-1])

        captions_batch = captions[:, 1:].contiguous().view(-1)
        captions_batch = vocabulary.map_to_trainset_ids(captions_batch)
        captions_batch = captions_batch.to(config.DEVICE)

        loss = loss_fn(
            output,
            captions_batch
        )

        optimizer.zero_grad()
        loss.backward()

        if config.GRAD_CLIP:
            clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        loader.set_postfix(loss=loss.item())

        losses.append(loss.item())

        val_loss = val_loss_epoch(
            test_loader,
            model,
            loss_fn,
            vocabulary
        )

        if scheduler is not None:
            scheduler.step(val_loss)

    if config.SAVE_MODEL:
        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': np.mean(losses)
        })

    print(f'Epoch[{epoch}]: Train Loss {np.mean(losses)}, Val Loss {val_loss}')


def val_loss_epoch(loader, model, loss_fn, vocabulary):
    model.eval()

    img, captions, masks = next(iter(loader))

    img = img.to(config.DEVICE)
    captions = captions.to(config.DEVICE)
    masks = masks.to(config.DEVICE)

    output = model(img, captions, masks)

    output = output.contiguous().view(-1, output.shape[-1])

    captions_batch = captions[:, 1:].contiguous().view(-1)
    captions_batch = vocabulary.map_to_trainset_ids(captions_batch)
    captions_batch = captions_batch.to(config.DEVICE)

    loss = loss_fn(
        output,
        captions_batch
    )

    return loss.item()


def main():
    all_dataset = utils.load_dataset()

    train_dataset, test_dataset = utils.train_test_split(dataset=all_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
        shuffle=True,
        collate_fn=CollateDataset(),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
        shuffle=False,
        collate_fn=CollateDataset(),
    )

    model = utils.get_model_instance(train_dataset)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.98))
    loss_fn = nn.CrossEntropyLoss(ignore_index=all_dataset.vocab.ids_to_ids_adj[all_dataset.vocab.tokens_to_ids['[PAD]']])

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=10, verbose=True)

    starting_epoch = 1

    if utils.can_load_checkpoint():
        starting_epoch = utils.load_checkpoint(model, optimizer)

    for epoch in range(starting_epoch, config.EPOCHS):
        train_epoch(
            train_loader,
            test_loader,
            model,
            optimizer,
            loss_fn,
            all_dataset.vocab,
            scheduler,
            epoch
        )


if __name__ == '__main__':
    main()
