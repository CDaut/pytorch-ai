import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from tqdm import tqdm
import wandb

wandb.init(project='pytorch_ai')
train = datasets.MNIST('./datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('./datasets', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))

trainset = torch.utils.data.DataLoader(train, batch_size=200, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 120)
        self.fc3 = nn.Linear(120, 120)
        self.fc4 = nn.Linear(120, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


net = Net()
wandb.watch(net)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('runnning on %s' % device)

net = net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(200):  # 10 full passes over the data
    for data in tqdm(trainset):  # `data` is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
        X = X.to(device)
        output = net(X.view(-1, 784))  # pass in the reshaped batch (recall they are 28x28 atm)
        output = output.cpu()
        loss = loss_function(output, y)  # calc and grab the loss value
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
        wandb.log({'loss': loss})
    net = net.cpu()
    torch.save(net, './nets/net_gpu_large_batch_' + str(epoch) + ".pt")
    net = net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testset:
            X, y = data
            X = X.to(device)
            output = net(X.view(-1, 784))
            output = output.cpu()
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    wandb.log({'test_accuracy': correct / total})
    print("Accuracy: ", round(correct / total, 3))
    wandb.log({'epoch': epoch})
