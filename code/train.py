import torch
from torch import optim
from torch import nn

from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import SimilarImagesDataset
from model import IMEncoder, IMDecoder

from tqdm import tqdm



def train_step(encoder, decoder, train_loader, lossfn, opt, device):
    encoder.train()
    decoder.train()

    for batch_idx, (train_img, target_img) in enumerate(train_loader):

        train_img = train_img.to(device)
        target_img = target_img.to(device)

        opt.zero_grad()

        enc_output = encoder(train_img)
        dec_output = decoder(enc_output)

        loss = lossfn(dec_output, target_img)
        loss.backward()
        print(f"train itr {batch_idx} ... loss {loss}")

        opt.step()

    return loss.item()


def validation_step(encoder, decoder, val_loader, lossfn, device):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, (val_img, target_img) in enumerate(val_loader):
            val_img = val_img.to(device)
            target_img = target_img.to(device)

            enc_output = encoder(val_img)
            dec_output = decoder(enc_output)

            loss = lossfn(dec_output, target_img)
            loss.backward()
            print(f"val itr {batch_idx} ... loss {loss}")

    return loss.item()


# Run the training on GPU
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# Normalize and convert to tensor
transforms = T.Compose([T.ToTensor()])

# Load the dataset
data = SimilarImagesDataset('data/similarity_test_examples/', transform=transforms)

# Train - val split
train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])
train_loader = DataLoader(train_data, batch_size=1)
val_loader = DataLoader(val_data, batch_size=1)

loss_fn = nn.MSELoss()

# model
encoder = IMEncoder(in_c=3, kernel=(3, 3), padding=(1, 1))
decoder = IMDecoder(out_c=3, kernel=(2, 2), stride=(2, 2))

# optimizer
autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
print(autoencoder_params)
opt = optim.Adam(autoencoder_params, lr=1e-3)

# train
epochs = 10
max_loss = 9999999

for epoch in tqdm(range(epochs)):

    train_loss = train_step(
        encoder, decoder, train_loader, loss_fn, opt, device)
    val_loss = validation_step(encoder, decoder, train_loader, loss_fn, device)

    print("Epoch {} || Training Loss {} || Val Loss {}".format(
        epoch, train_loss, val_loss))

    # save best model
    if val_loss < max_loss:
        max_loss = val_loss
        print("Validation Loss decreased, saving new best model")
        torch.save(encoder.state_dict(), "models/encoder_model.pt")
        torch.save(decoder.state_dict(), "models/decoder_model.pt")
