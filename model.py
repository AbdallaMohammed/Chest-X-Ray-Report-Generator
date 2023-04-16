import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import config


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        self.resnet152 = models.resnet152(weights='DEFAULT')

        for param in self.resnet152.parameters():
            param.requires_grad_(False)

        self.resnet152.fc = nn.Linear(self.resnet152.fc.in_features, embed_size)


    def forward(self, images):
        features = self.resnet152(images)

        return F.dropout(F.relu(features), p=0.5)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bias=True, num_layers=1)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeds = F.dropout(self.embedding(captions), p=0.25)
        embeds = torch.cat((features.unsqueeze(0), embeds), dim=0)
        
        h, _ = self.lstm(embeds)

        return self.fc(h)


class EncoderDecoderNet(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(EncoderDecoderNet, self).__init__()

        self.encoder = EncoderCNN(embed_size=embed_size)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        output = self.decoder(features, captions)

        return output

    def generate_caption(self, image, vocabulary, max_length=25):
        caption = []

        with torch.no_grad():
            features = self.encoder(image)
            states = None

            for _ in range(max_length):
                h, states = self.decoder.lstm(features.unsqueeze(0), states)
                
                output = self.decoder.fc(h.squeeze(0))
                predicted = output.argmax(1)

                caption.append(predicted.item())

                features = self.decoder.embedding(predicted)

                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break

        return [vocabulary.itos[idx] for idx in caption]
