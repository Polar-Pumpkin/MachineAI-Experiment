import os
import time
from typing import List

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.data.wn18 import WN18Definitions, WN18Dataset
from src.net import TransE
from src.util.general import poll

dimension = 50
margin = 1.0
norm = 1
c = 1.0
lr = 0.01

output_root = os.path.join('.', 'runs')
output_path = os.path.join(output_root, time.strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

dataset_root = os.path.join('.', 'datasets')
dataset_path = os.path.join(dataset_root, 'WN18')
definitions = WN18Definitions(os.path.join(dataset_path, 'wordnet-mlj12-definitions.txt'))
train_set = WN18Dataset(dataset_path, 'train', definitions)
validate_set = WN18Dataset(dataset_path, 'valid', definitions)
test_set = WN18Dataset(dataset_path, 'test', definitions)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = TransE(len(definitions.entities), len(definitions.relations), dimension, margin, norm, c).to(device=device)
optimizer = optim.Adam(net.parameters(), lr=lr)


def train(epoches: int = 50, batch_size: int = 100):
    train_batches = int(len(train_set) / batch_size)
    validate_batches = int(len(validate_set) / batch_size) + 1
    print(f'Batches: {train_batches}/{validate_batches}')

    train_losses: List[float] = []
    validate_losses: List[float] = []
    for epoch in range(epoches):
        print(f'===== Epoch {epoch}/{epoches}')
        timestamp = time.time()
        train_loss = 0.0
        validate_loss = 0.0
        # Normalise the embedding of the entities to 1

        net.train()
        for _ in tqdm(range(train_batches), desc='Train'):
            triple, corrupted_triple = poll(batch_size, train_set, definitions, device)
            optimizer.zero_grad()
            loss = net(triple, corrupted_triple)

            train_loss += loss
            loss.backward()
            optimizer.step()

        net.eval()
        for _ in tqdm(range(validate_batches), desc='Validate'):
            triple, corrupted_triple = poll(batch_size, validate_set, definitions, device)
            validate_loss += net(triple, corrupted_triple)

        mean_train_loss = train_loss / train_batches
        mean_validate_loss = validate_loss / validate_batches
        print(f'Losses: {mean_train_loss}/{mean_validate_loss}')

        if epoch % 5 == 0 or epoch == epoches:
            filename = 'Epoch({})-{:.3f}-{:.3f}.pth'.format(epoch, mean_train_loss, mean_validate_loss)
            print(f'保存阶段性模型至 {filename}')
            torch.save(net.state_dict(), os.path.join(output_path, filename))

        if len(validate_losses) <= 0 or mean_validate_loss <= min(validate_losses):
            filename = 'best.pth'
            print(f'保存性能最好的模型至 {filename}')
            torch.save(net.state_dict(), os.path.join(output_path, filename))

        filename = 'latest.pth'
        torch.save(net.state_dict(), os.path.join(output_path, filename))

        now = time.time()
        print(f'{round(now - timestamp, 3)}s elpased')
        train_losses.append(mean_train_loss)
        validate_losses.append(mean_validate_loss)

    # visualize the loss as the network trained
    figure = plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train losses')
    plt.plot(range(1, len(validate_losses) + 1), validate_losses, label='Validate losses')

    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.xlim(0, len(train_losses) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.title('TransE training losses')

    figure.savefig(os.path.join(output_path, 'Losses.png'), bbox_inches='tight')
    with open(os.path.join(output_path, 'losses.txt'), 'w') as file:
        file.writelines('\n'.join(map(str, train_losses)))
    with open(os.path.join(output_path, 'validate_losses.txt'), 'w') as file:
        file.writelines('\n'.join(map(str, validate_losses)))


if __name__ == '__main__':
    train()
