import os
import torchaudio
import torch
import noisereduce as nr

files = sorted(os.listdir('chunks'))
i = 0
batch_size = 8
while i < len(files):
    batch = []
    for j in range(batch_size):
        data, fs = torchaudio.load('chunks/' + files[i+j])
        batch.append(data)
    
    batch = torch.cat(batch, dim=0)
    batch = nr.reduce_noise(y=batch, sr=32000, n_jobs=-1)
    for j in range(batch_size):
        data, fs = torchaudio.load('chunks/' + files[i+j])
        torchaudio.save('denoised/' + files[i+j], batch[j, :], 32000)
    i =+ batch_size


