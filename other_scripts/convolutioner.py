import torch
import PIL
import numpy
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


def frobdot(mat1, mat2):
    return mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[0][1] + mat1[0][2] * mat2[0][2] + mat1[1][0] * mat2[1][0] + \
           mat1[1][1] * mat2[1][1] + mat1[1][2] * mat2[1][2] + mat1[2][0] * mat2[2][0] + mat1[2][1] * mat2[2][1] + \
           mat1[2][2] * mat2[2][2]


data = datasets.MNIST('../datasets', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))

loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
examples = enumerate(loader)
batch_idx, (example_data, example_targets) = next(examples)

raw_img = example_data[0][0]


def convolve(raw_img, kernel):
    new_img = numpy.zeros((28, 28))
    for y, line in enumerate(raw_img):
        for x, value in enumerate(line):
            tempmat = numpy.zeros((3, 3))
            possible_positions = [[y + 1, x - 1], [y + 1, x], [y + 1, x + 1], [y, x - 1], [y, x], [y, x + 1],
                                  [y - 1, x - y], [y - 1, x], [y - 1, x - 1]]
            n = -1
            for i, pos in enumerate(possible_positions):
                if i % 3 is 0:
                    n += 1

                if (pos[0] < 0) or (pos[0] > 27) or (pos[1] < 0) or (pos[1] > 27):
                    tempmat[n][i % 3] = 0
                else:
                    tempmat[n][i % 3] = raw_img[pos[0]][pos[1]]
            new_img[y][x] = frobdot(tempmat, kernel)

    return new_img


def maxpool(raw_image):
    new_img = numpy.zeros((14, 14))
    new_y = -1
    for y in range(0, 28, 2):
        new_x = -1
        new_y += 1
        for x in range(0, 28, 2):
            new_x += 1
            new_img[new_y][new_x] = max(
                [raw_image[y][x], raw_image[y][x + 1], raw_image[y + 1][x], raw_image[y + 1][x + 1]])
    return new_img


def avgpool(raw_image):
    new_img = numpy.zeros((14, 14))
    new_y = -1
    for y in range(0, 28, 2):
        new_x = -1
        new_y += 1
        for x in range(0, 28, 2):
            new_x += 1
            new_img[new_y][new_x] = sum(
                [raw_image[y][x], raw_image[y][x + 1], raw_image[y + 1][x], raw_image[y + 1][x + 1]]) / 4
    return new_img


kernel = numpy.array([[-1, -1, -1],
                      [1, 1, 1],
                      [0, 0, 0]])

plt.xticks([])
plt.yticks([])
plt.imshow(avgpool(convolve(raw_img, kernel)), cmap='gray')
plt.show()
