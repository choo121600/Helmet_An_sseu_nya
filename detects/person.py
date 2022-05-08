import cv2
import torch
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots2 import plot_one_box
from utils.torch_utils import select_device
import imutils
import time
from settings import *

prev_per = []
class Person:
    def __init__(self):
        self.body_state = "Stop"
        self.per_li = []
        self.per_x_li = []
        self.per_y_li = []

    def detect_person(self, img0, xyxy, detect_name, draw_color):
        global prev_per
        if detect_name == 'Body':
            person_c1, person_c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            person_x, person_y = person_c1[0] + (person_c2[0] - person_c1[0]) / 2, person_c1[1] + (person_c2[0] - person_c1[0]) / 2
            self.per_li.append([person_x, person_y])
            person_x1, person_x2 = person_c1[0], person_c2[0]
            person_y1, person_y2 = person_c1[1], person_c2[1]
            self.per_x_li.append([person_x1, person_x2])
            self.per_y_li.append([person_y1, person_y2])

            # Check person is moving
            if prev_per != []:
                for per in self.per_li:
                    for prev in prev_per:
                        if TRACKING_SPEED[0] < abs(prev[0] - per[0]) < TRACKING_SPEED[1] and TRACKING_SPEED[0] < abs(prev[1] - per[1]) < TRACKING_SPEED[1]:
                            self.body_state = "Move"
            prev_per = self.per_li

            label = "Body "+ self.body_state
            plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)