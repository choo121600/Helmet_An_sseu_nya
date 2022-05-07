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

prev_pho = []
class Bike_fin:
    def __init__(self):
        self.state = "Stop"
        self.pho_li = []

    def detect_bike(self, img0, xyxy, detect_name, draw_color, detect_conf):
        global prev_pho
        if detect_name == 'cell phone':
            bi_c1, bi_c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            bi_x1, bi_x2 = bi_c1[0], bi_c2[0]
            self.pho_li.append([bi_x1, bi_x2])

            # Check Bike is moving
            if prev_pho != []:
                for pho in self.pho_li:
                    for prev in prev_pho:
                        if TRACKING_SPEED[0] < abs(prev[0] - pho[0]) < TRACKING_SPEED[1] and TRACKING_SPEED[0] < abs(prev[1] - pho[1]) < TRACKING_SPEED[1]:
                            self.state = "Move"
            prev_pho = self.pho_li

            if self.state == "Move":
                label = detect_name +" "+ str(round(detect_conf, 2)) + self.state
                plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)
                # self.track_bike()
            else:
                label = detect_name +" "+ str(round(detect_conf, 2)) + self.state
                plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)