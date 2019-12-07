from .main import Board

end = False
board = [0, 0, 0,
         0, 0, 0,
         0, 0, 0]
b = Board(barray=board)
while not end:
    n = int(input('Enter field: '))
    b.board[n] = -1
    b.refresh()
    print(b)
