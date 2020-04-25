# noinspection PyUnresolvedReferences
from main import Board


def loadboards():
    boards = {}
    with open('boards.bds', 'r') as infile:
        for line in infile.readlines():
            readline = line.split('|')
            boards[readline[0]] = readline[1].replace('\n', '')
    return boards


end = False
board = [0, 0, 0,
         0, 0, 0,
         0, 0, 0]
b = Board(barray=board)
boards = loadboards()

pos = boards[b.printversion]
b.board[int(pos)] = 1
b.refresh()
print(b)
print('----------------------')

while not end:
    if b.full and b.winner is 0:
        print('Draw!')
        end = True
    else:
        n = int(input('Enter field: '))
        if n > 8:
            print('haha.')
        else:
            if b.board[n] is not 0:
                print('This field is already used.')
            else:
                b.board[n] = -1
                b.refresh()
                print(b)
                print('----------------------')
                pos = boards[b.printversion]
                b.board[int(pos)] = 1
                b.refresh()
                print(b)
                print('----------------------')
                if b.winner is not 0:
                    print(str(b.winner) + ' won!')
                    end = True
