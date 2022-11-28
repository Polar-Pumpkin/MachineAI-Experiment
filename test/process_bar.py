import time

from tqdm import tqdm

length = 100
bar = tqdm(range(100), desc='Process', mininterval=0.3)
bar.set_postfix(**{'index': '?'})

for index in bar:
    bar.set_postfix(**{'index': index})
    if index == 50:
        print(f'Index == 50 !!!')
    time.sleep(0.2)
