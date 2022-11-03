import torch

with open('memory_summary.txt', 'w') as file:
    file.write(torch.cuda.memory_summary())