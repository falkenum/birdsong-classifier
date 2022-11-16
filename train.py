import torchaudio
from torchaudio.transforms import MelSpectrogram
import pytorch_lightning as pl
from torchinfo import summary
# from torch.utils.data.dataset import random_split
import torch
from torch import nn, optim
from torch.utils.data.dataloader import MapDataPipe, DataLoader
from torch.utils.data.dataset import random_split
from pathlib import Path
import os

SR = 32000
BATCH_SIZE = 2

class BirdsongDataset(MapDataPipe):
    def __init__(self, samplesdir: str) -> None:
        self.samplesdir = samplesdir
        self.filenames = sorted(os.listdir(samplesdir))

        class_names = set()

        for file in self.filenames:
            class_names.add(file[0])
        
        self.class_name_to_idx = {name: i for i, name in enumerate(sorted(list(class_names)))}

        super().__init__()

    def __getitem__(self, index):
        data, fs = torchaudio.load(f'{self.samplesdir}/{self.filenames[index]}')
        assert(fs == SR)
        return data, self.class_name_to_idx[self.filenames[index][0]] #tensor, label
    
    def __len__(self):
        return len(self.filenames)

dataset = BirdsongDataset('chunks')
traindata, testdata= random_split(dataset, (0.9, 0.1))
traindata, valdata = random_split(traindata, (0.8, 0.2))

train_dataloader = DataLoader(traindata, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(valdata, batch_size=100)
# test_dataloader = DataLoader(testdata, batch_size=BATCH_SIZE)

sample_batch = torch.cat([dataset[i][0][None, :] for i in range(BATCH_SIZE)], dim=0)

conv = nn.Sequential(
    MelSpectrogram(sample_rate=SR, n_fft=1024),
    nn.Conv2d(1, 32, kernel_size=(3, 7), padding='same'),
    nn.MaxPool2d(kernel_size=(1, 4)),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=(3, 5), padding='same'),
    nn.MaxPool2d(kernel_size=(1, 4)),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
    nn.MaxPool2d(kernel_size=(2, 3)),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
    nn.MaxPool2d(kernel_size=(4, 4)),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
    nn.MaxPool2d(kernel_size=(4, 4)),
    nn.ReLU(),
    nn.Flatten(),
)
conv_outshape = conv(sample_batch).shape[1]

dense = nn.Linear(in_features=conv_outshape, out_features=len(dataset.class_name_to_idx.keys()))
model = nn.Sequential(conv, dense)

print(summary(model, input_size=sample_batch.shape))

class BirdsongClassifer(pl.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.classifier = model
    
    def training_step(self, batch, batch_idx):
        data, label = batch
        data = self.classifier(data)
        data = torch.reshape(data, (data.shape[0], -1))
        loss = nn.functional.cross_entropy(data, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        data = self.classifier(data)
        data = torch.reshape(data, (data.shape[0], -1))
        loss = nn.functional.cross_entropy(data, label)
        self.log('val_loss', loss)

        data = torch.softmax(data, dim=1).round()
        label = nn.functional.one_hot(label, num_classes=len(dataset.class_name_to_idx))
        correct = torch.count_nonzero(torch.all(data == label, dim=1))
        # print(correct)
        self.log('percent_correct', correct / 100)
        
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


classifier = BirdsongClassifer(model)


trainer = pl.Trainer(max_epochs=10000, accelerator='gpu', check_val_every_n_epoch=5)
trainer.fit(model=classifier, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, )

# trainer.test()

# batch = next(iter(dataloader)) classifier.training_step(batch, 0)
    

