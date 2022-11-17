import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
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
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 100

class BirdsongDataset(MapDataPipe):
    def __init__(self, samplesdir: str) -> None:
        self.samplesdir = samplesdir
        self.filenames = sorted(os.listdir(samplesdir))

        class_names = set()

        for file in self.filenames:
            class_names.add(file[:file.index('_')])
        
        self.class_name_to_idx = {name: i for i, name in enumerate(sorted(list(class_names)))}

        super().__init__()

    def __getitem__(self, index):
        data, fs = torchaudio.load(f'{self.samplesdir}/{self.filenames[index]}')
        assert(fs == SR)
        filename = self.filenames[index]
        return data, self.class_name_to_idx[filename[:filename.index('_')]] #tensor, label
    
    def __len__(self):
        return len(self.filenames)

dataset = BirdsongDataset('denoised')
torch.manual_seed(0)
traindata, testdata= random_split(dataset, (0.8, 0.2))
traindata, valdata = random_split(traindata, (0.9, 0.1))

train_dataloader = DataLoader(traindata, batch_size=TRAIN_BATCH_SIZE, num_workers=12)
val_dataloader = DataLoader(valdata, batch_size=VAL_BATCH_SIZE, num_workers=12)
# test_dataloader = DataLoader(testdata, batch_size=BATCH_SIZE)

sample_batch = torch.cat([dataset[i][0][None, :] for i in range(TRAIN_BATCH_SIZE)], dim=0)

process = nn.Sequential(
    MelSpectrogram(sample_rate=SR, n_fft=1024),
    AmplitudeToDB(),
)

conv = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=(3, 7)),
    nn.MaxPool2d(kernel_size=(1, 4)),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=(3, 5)),
    nn.MaxPool2d(kernel_size=(1, 4)),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=(3, 3)),
    nn.MaxPool2d(kernel_size=(4, 4)),
    nn.ReLU(),
    nn.Flatten(),
)
conv_outshape = conv(process(sample_batch)).shape[1]

dense = nn.Sequential(
    nn.Linear(in_features=conv_outshape, out_features=conv_outshape//128),
    nn.Linear(in_features=conv_outshape//128, out_features=len(dataset.class_name_to_idx)),
)
model = nn.Sequential(process, conv, dense)

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

        data = torch.argmax(torch.softmax(data, dim=1), dim=1)
        correct = torch.count_nonzero(data == label)
        self.log('percent_correct', correct / VAL_BATCH_SIZE * 100)
        
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


classifier = BirdsongClassifer(model)
trainer = pl.Trainer(max_epochs=10000, accelerator='gpu', check_val_every_n_epoch=3)
trainer.fit(model=classifier, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

