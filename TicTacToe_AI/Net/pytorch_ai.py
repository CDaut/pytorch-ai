import random
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def to_set(raw_list):
    out_set = []
    for line in tqdm(raw_list):
        line = line.replace('\n', '')
        raw_board, raw_label = line.split('|')[0], line.split('|')[1]

        # convert string label to tensor
        label = torch.zeros([1, 1], dtype=torch.long)
        if not (int(raw_label) is -1):
            label[0][0] = int(raw_label)
        else:
            label[0][0] = 9

        # convert board to tensor
        raw_board = raw_board.split(',')
        board = torch.zeros([1, 9])
        for n, block in enumerate(raw_board):
            if int(block) is -1:
                board[0][n] = 0
            elif int(block) is 0:
                board[0][n] = 0.5
            elif int(block) is 1:
                board[0][n] = 1

        out_set.append((board, label))

    return out_set


def buildsets():
    with open('boards.bds', 'r') as infile:
        print('Loading file...')
        alllines = infile.readlines()
        print(len(alllines))
        random.shuffle(alllines)

        print('Generating testset...')
        testset = to_set(alllines[0:10000])

        print('Generating trainset...')
        trainset = to_set(alllines[10001:200000])

    return trainset, testset


def testnet(net, testset):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, label in testset:
            output = net(X)
            if torch.argmax(output) == label[0]:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct / total, 3))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 9)
        self.fc2 = nn.Linear(9, 20)
        self.fc3 = nn.Linear(20, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)

trainset, testset = buildsets()

for epoch in range(100):
    print('Epoch: ' + str(epoch))
    for X, label in tqdm(trainset):
        net.zero_grad()
        output = net(X)
        loss = F.nll_loss(output.view(1, 10), label[0])
        loss.backward()
        optimizer.step()

    print(loss)
    torch.save(net, './nets/net_' + str(epoch) + '.pt')
    testnet(net, testset)
