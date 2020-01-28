import random
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

wandb.init(project="tictactoe")

BATCH_SIZE = 250


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


def to_batched_set(raw_list):
    counter = 0
    out_set = []
    boardtensor = torch.zeros((BATCH_SIZE, 9))
    labeltensor = torch.zeros(BATCH_SIZE, dtype=torch.long)
    for line in tqdm(raw_list):
        line = line.replace('\n', '')
        raw_board, raw_label = line.split('|')[0], line.split('|')[1]

        if not (int(raw_label) is -1):
            labeltensor[counter] = int(raw_label)
        else:
            labeltensor[counter] = 9

        raw_board = raw_board.split(',')
        for n, block in enumerate(raw_board):
            if int(block) is -1:
                boardtensor[counter][n] = 0
            elif int(block) is 0:
                boardtensor[counter][n] = 0.5
            elif int(block) is 1:
                boardtensor[counter][n] = 1

        if counter == (BATCH_SIZE - 1):
            out_set.append([boardtensor, labeltensor])
            boardtensor = torch.zeros((BATCH_SIZE, 9))
            labeltensor = torch.zeros(BATCH_SIZE, dtype=torch.long)
            counter = 0
        else:
            counter += 1

    return out_set


def buildsets():
    with open('boards.bds', 'r') as infile:
        print('Loading file...')
        alllines = infile.readlines()
        print(len(alllines))
        random.shuffle(alllines)

        print('Generating testset...')
        testset = to_batched_set(alllines[0:10000])

        print('Generating trainset...')
        trainset = to_batched_set(alllines[10001:])

    return trainset, testset


def testnet(net, testset, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, label in testset:
            X = X.to(device)
            output = net(X)
            output = output.cpu()
            if torch.argmax(output) == label[0]:
                correct += 1
            total += 1
    wandb.log({'test_accuracy': correct / total})
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('running on %s' % device)

# net = torch.load('./nets/net_3.pt')

net = Net()
wandb.watch(net)

net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

trainset, testset = buildsets()

for epoch in range(300):
    print('Epoch: ' + str(epoch))
    wandb.log({'epoch': epoch})
    for X, label in tqdm(trainset):
        net.zero_grad()
        X = X.to(device)
        output = net(X)
        output = output.cpu()
        loss = loss_function(output.view(-1, 10), label)
        loss.backward()
        optimizer.step()
        wandb.log({'loss': loss})

    net = net.cpu()
    torch.save(net, './nets/gpunets/net_' + str(epoch) + '.pt')
    net = net.to(device)
    testnet(net, testset, device)
