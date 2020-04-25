import pygame
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def transform_to_rgb(value):
    if value < 0 or value > 1:
        return tuple((255, 255, 255))
    else:
        return tuple((value * 255, value * 255, value * 255))


def update_screen_from_array(array, screen):
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            x_transformed = x * 30
            y_transformed = y * 30 + 161

            color = transform_to_rgb(array[x, y])
            pygame.draw.rect(screen, color, (x_transformed, y_transformed, 30, 30), 0)


########################
HARDNESS = 0.7
########################

pygame.init()
screen = pygame.display.set_mode((28 * 30, 1000))
clock = pygame.time.Clock()
stopFlag = False
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
img_array = np.zeros((28, 28))

font = pygame.font.SysFont('linuxbiolinum', 80)

net = torch.load('./nets/net_gpu_large_batch_199.pt')

while not stopFlag:

    # get events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stopFlag = True

    pygame.draw.line(screen, WHITE, [0, 160], [pygame.display.get_surface().get_size()[0], 160], 1)  # Trennlinie

    if pygame.mouse.get_pressed()[0] == 1:
        pos = pygame.mouse.get_pos()

        # transform mouse positions to array indices
        x_transformed = math.floor((pos[0] / 30))
        y_transformed = math.floor((pos[1] - 161) / 30)
        if x_transformed < 0:
            x_transformed = 0
        if y_transformed < 0:
            y_transformed = 0
        else:
            pass
        coords = [(x_transformed, y_transformed), (x_transformed + 1, y_transformed),
                  (x_transformed - 1, y_transformed), (x_transformed, y_transformed + 1),
                  (x_transformed, y_transformed - 1)]

        for coord in coords:
            if not coord[0] < 0 and not coord[0] > 27 and not coord[1] < 0 and not coord[1] > 27:
                img_array[coord[0], coord[1]] += HARDNESS

    # Clear image on right mouse button press
    if pygame.mouse.get_pressed()[2] == 1:
        img_array = np.zeros((28, 28))

    update_screen_from_array(img_array, screen)
    # Draw vertical lines
    for i in range(30, 28 * 30, 30):
        pygame.draw.line(screen, WHITE, [i, 161], [i, 1000], 1)

    # Draw horizontal lines
    for i in range(189, 1000, 30):
        pygame.draw.line(screen, WHITE, [0, i], [28 * 30, i], 1)

    pygame.display.flip()
    clock.tick(60)
    np.rot90(img_array, k=1)
    tensor = torch.from_numpy(img_array).view(1, 28 * 28).float()
    with torch.no_grad():
        prediction = torch.argmax(net(tensor))

        text = font.render('Prediction: ' + str(prediction.item()), True, WHITE, BLACK)
        textRect = text.get_rect()
        textRect.center = (420, 80)
        screen.blit(text, textRect)

pygame.quit()
