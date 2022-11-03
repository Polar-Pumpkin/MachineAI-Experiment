import torch

with open('memory_summary.txt', 'w') as file:
    file.write(torch.cuda.memory_summary())

with open('memory_stats.txt', 'w') as file:
    file.write(str(torch.cuda.memory_stats()))
