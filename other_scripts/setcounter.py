from tqdm import tqdm
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

data = datasets.MNIST('../datasets', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))

loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

for data in loader:
    tensor = data[0].view([28, 28])
    plt.imshow(tensor)
    plt.show()

