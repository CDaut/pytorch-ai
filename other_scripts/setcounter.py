from tqdm import tqdm
import torch
from torchvision import transforms, datasets

data = datasets.MNIST('../datasets', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))

loader = torch.utils.data.DataLoader(data, batch_size=15, shuffle=False)
set = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}

for data in loader:
    print(data[1].shape)

for _, label in tqdm(loader):
    set[str(label[0].item())] += 1

print(set)

num = 0
for x in set:
    num += set[x]
print(num)

for x in set:
    set[x] /= num
    set[x] *= 100
print(set)
