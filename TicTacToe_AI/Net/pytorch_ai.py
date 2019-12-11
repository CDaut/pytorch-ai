import random
import torch
from tqdm import tqdm


def to_set(raw_list):
    out_set = []
    for line in tqdm(raw_list):
        line = line.replace('\n', '')
        raw_board, raw_label = line.split('|')[0], line.split('|')[1]

        # convert string label to tensor
        label = torch.zeros([1, 9])
        if not (int(raw_label) is -1):
            label[0][int(raw_label)] = 1

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


with open('boards.bds', 'r') as infile:
    print('Loading file...')
    alllines = infile.readlines()
    random.shuffle(alllines)

    print('Generating testset...')
    testset = to_set(alllines[0:50000])

    print('Generating trainset...')
    trainset = to_set(alllines[50001:])
