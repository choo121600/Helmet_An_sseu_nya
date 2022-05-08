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

prev_bike = []
class Bike:
    def __init__(self):
        self.bike_state = "Stop"
        self.bike_li = []
        self.bike_x_li = []
        self.bike_y_li = []

    def detect_bike(self, img0, xyxy, detect_name, draw_color):
        global prev_bike
        if detect_name == 'Autobike' or detect_name == 'Bicycle':
            bi_c1, bi_c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            bi_xmid, bi_ymid = bi_c1[0] + (bi_c2[0] - bi_c1[0]) / 2, bi_c1[1] + (bi_c2[0] - bi_c1[0]) / 2
            bi_x1, bi_x2 = bi_c1[0], bi_c2[0]
            bi_y1, bi_y2 = bi_c1[1], bi_c2[1]
            self.bike_x_li.append([bi_x1, bi_x2])
            self.bike_y_li.append([bi_y1, bi_y2])
            self.bike_li.append([bi_xmid, bi_ymid])

            # Check Bike is moving
            if prev_bike != []:
                for bike in self.bike_li:
                    for prev in prev_bike:
                        if TRACKING_SPEED[0] < abs(prev[0] - bike[0]) < TRACKING_SPEED[1] and TRACKING_SPEED[0] < abs(prev[1] - bike[1]) < TRACKING_SPEED[1]:
                            self.bike_state = "Move"
            prev_bike = self.bike_li

            label = "Bike "+ self.bike_state
            plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)
