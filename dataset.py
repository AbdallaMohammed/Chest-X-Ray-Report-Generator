import os
import torch
import config
import utils
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from vocabulary import Vocab
from torch.utils.data import Dataset, DataLoader


class XRayDataset(Dataset):
    def __init__(self, root, transform=None, raw_caption=False):
        self.root = root
        self.transform = transform
        self.raw_caption = raw_caption

        self.captions = []
        self.imgs = []

        for file in os.listdir(os.path.join(self.root, 'reports')):
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(self.root, 'reports', file))

                frontal_img = ''
                findings = tree.find(".//AbstractText[@Label='FINDINGS']").text

                if findings is None:
                    continue

                for x in tree.findall('parentImage'):
                    if frontal_img != '':
                        break

                    img = x.attrib['id']
                    img = os.path.join(config.IMAGES_DATASET, f'{img}.png')

                    frontal_img = img

                if frontal_img == '':
                    continue
                
                self.captions.append(utils.normalize_text(findings))
                self.imgs.append(frontal_img)

        self.vocab = Vocab(self.captions)

    def __getitem__(self, item):
        img = self.imgs[item]
        caption = self.captions[item]

        img = np.array(Image.open(img).convert('L'))
        img = np.expand_dims(img, axis=-1)
        img = img.repeat(3, axis=-1)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        if self.raw_caption:
            return img, caption

        return img, caption

    def __len__(self):
        return len(self.captions)

    def get_caption(self, item):
        return self.captions[item].split(' ')


class CollateDataset:
    def __init__(self):
        self.tokenizer = utils.load_bert_tokenizer()

    def __call__(self, batch):
        images, captions = zip(*batch)

        images = torch.stack(images, 0)

        targets = self.tokenizer(
            captions,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt'
        )

        targets = targets.input_ids
        masks = torch.ones_like(targets)

        return images, targets, masks


if __name__ == '__main__':
    all_dataset = XRayDataset(
        root=config.DATASET_PATH,
        transform=config.basic_transforms,
    )

    train_loader = DataLoader(
        dataset=all_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        shuffle=True,
        collate_fn=CollateDataset(),
    )

    print(len(all_dataset.vocab))

    for img, captions, masks in train_loader:
        captions = captions.view(-1).cpu().detach().numpy()
        print(len(all_dataset.vocab.tokenizer.convert_ids_to_tokens(captions)))
        # print(all_dataset.vocab.map_to_trainset_ids(captions_batch))
        break