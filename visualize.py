import numpy as np
import pandas as pd
import pygame
import glob
#from config import VisualizeConfig
import os
import time
import random

import csv

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

pygame.init()
size = (320,240)
pygame.display.set_caption("Data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
myfont = pygame.font.SysFont("comicsans", 25)

filenames = []
images_list = []

file_src = './center/'
true_angle=0
angle=0

for i in (sorted(os.listdir(file_src))):
    with open('test_steering_for_visualization.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0].split('/')[2]==i:
                true_angle=-3*float(row[1])
                angle= -3*float(row[2])
                print(true_angle-angle)
                # add image to screen
                #img = cv2.imread(filenames[i])
                #cv2.imshow("visualize",img)
                #cv2.waitKey(1)
        
        img = pygame.image.load(file_src + i)
        
        screen.blit(img, (0, 0))

        # draw steering wheel
        radius = 50
        pygame.draw.circle(screen, WHITE, [160, 120], radius, 2)
        pred_txt = myfont.render("   CNN  Angle:" + str(round(angle* 57.2958, 3)), 50, (255,0,0))
        true_txt = myfont.render("Human Angle:" + str(round(true_angle* 57.2958, 3)), 50, (0,255,0)) 
        screen.blit(pred_txt, (10, 30))
        screen.blit(true_txt, (10, 50))

        # draw cricle for true angle
        x = radius * np.cos(np.pi / 2 + true_angle)
        y = radius * np.sin(np.pi / 2 + true_angle)
        pygame.draw.circle(screen, GREEN, [160 + int(x), 120 - int(y)], 7)

        # draw cricle for predicted angle
        x = radius * np.cos(np.pi / 2 + angle)
        y = radius * np.sin(np.pi / 2 + angle)
        pygame.draw.circle(screen, RED, [160 + int(x), 120 - int(y)], 5)


        #pygame.display.update()
        pygame.display.flip()
        time.sleep(0.05)

