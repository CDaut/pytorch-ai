def boarddiff(b1, b2):
    sums = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, n in enumerate(b1.board):
        sums[i] = n + b2.board[i]

    for i, n in enumerate(sums):
        if (n is 1) or (n is -1):
            return i


class Board:
    def __init__(self, barray=None):
        self.board = barray  # 0: unset, -1: circle, 1: cross
        self.full = self.getfull()
        self.next = self.getnext()  # Cross always starts
        self.winner = self.getwinner()
        self.printversion = ','.join(str(x) for x in self.board)

    def __str__(self):
        retstr = ''
        for n, block in enumerate(self.board):
            if (n + 1) % 3 is not 0:
                retstr += str(block) + '|'
            else:
                retstr += str(block) + '\n'
        return retstr

    # check if the board is full or not
    def getfull(self):
        for block in self.board:
            if block is 0:  # if an unset block is found the board is not full
                return False
        return True

    # check who is next
    def getnext(self):
        if self.full:
            return None
        countdict = {0: 0, -1: 0, 1: 0}

        for block in self.board:
            countdict[block] += 1

        if (countdict[-1] > countdict[1]) or (countdict[-1] == countdict[1]):
            return 1  # cross always starts
        elif countdict[-1] < countdict[1]:
            return -1

    # check who has won
    def getwinner(self):
        # check rows
        if (sum(self.board[0:3]) is -3) or (sum(self.board[3:6]) is -3) or (sum(self.board[6:]) is -3):
            return -1
        elif (sum(self.board[0:3]) is 3) or (sum(self.board[3:6]) is 3) or (sum(self.board[6:]) is 3):
            return 1

        # check colums
        for i in range(3):
            if (self.board[0 + i] + self.board[3 + i] + self.board[6 + i]) is -3:
                return -1
            elif (self.board[0 + i] + self.board[3 + i] + self.board[6 + i]) is 3:
                return 1

        # main diagonal
        if (self.board[0] + self.board[4] + self.board[8]) is -3:
            return -1
        elif (self.board[0] + self.board[4] + self.board[8]) is 3:
            return 1

        # secondary diagonal
        elif (self.board[2] + self.board[4] + self.board[6]) is -3:
            return -1
        elif (self.board[2] + self.board[4] + self.board[6]) is 3:
            return 1

        return 0

    def refresh(self):
        self.next = self.getnext()
        self.full = self.getfull()
        self.winner = self.getwinner()
        self.printversion = ','.join(str(x) for x in self.board)

    # generate valid next boards
    def nextboards(self):
        if self.full:
            return []
        else:
            newboards = []
            for i, block in enumerate(self.board):
                if block is 0:
                    arr = [0, 0, 0, 0, 0, 0, 0, 0, 0]

                    for n in range(9):
                        arr[n] = self.board[n]

                    b = Board(barray=arr)
                    b.board[i] = self.next
                    b.refresh()
                    newboards.append(b)

            return newboards


class Node:
    def __init__(self, board):
        self.board = board
        self.children = self.genchildren()
        self.worth = self.calcworth()
        self.nextbest = self.getnextbest()
        print('generated node of board: \n' + str(self.board))
        print('nextbest: ' + str(self.nextbest))
        with open('boards.bds', 'a') as of:
            of.write(self.board.printversion + '|' + str(self.nextbest) + '\n')

    # generate children of node
    def genchildren(self):
        children = []
        for nextboard in self.board.nextboards():
            n = Node(nextboard)
            children.append(n)
        return children

    def calcworth(self):
        worth = 0
        for b in self.children:
            worth += b.board.winner
        return worth

    def getnextbest(self):
        if not self.board.full:
            highest = -9999999999
            highchild = None
            for child in self.children:
                if child.worth > highest:
                    highest = child.worth
                    highchild = child
            return boarddiff(self.board, highchild.board)
        else:
            return -1  # returns -1 as next best index if board is full


if __name__ == '__main__':
    b = Board(barray=[0, 0, 0,
                      0, 0, 0,
                      0, 0, 0])

    n = Node(b)
