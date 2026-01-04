from tqdm import tqdm
import time


apples = 1
tree = range(0, 180000)

for a in tqdm(tree):
      apples += 1
      time.sleep(0.0001)