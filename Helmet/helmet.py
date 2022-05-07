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
from Helmet.helmet import Helmet


class Helmet_fin:
    def __init__(self):
        self.state = "Stop"
    
    def detect_helmet(self, img0, xyxy, detect_name, draw_color, detect_conf, per_x_li, per_y_li):
        global prev_per
        if detect_name == 'helmet':
            helmet_c1, helmet_c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            helmet_x_mid, helmet_y_mid = int((helmet_c1[0] + helmet_c2[0]) / 2), int((helmet_c1[1] + helmet_c2[1]) / 2)

            if per_x_li != [] and per_y_li != []:
                for per_x in per_x_li:
                    for per_y in per_y_li:
                        if per_x[0] < helmet_x_mid < per_x[1] and per_y[0] < helmet_y_mid < per_y[1]:
                            print("Helmet is in the person")
                            label = detect_name +" "+ str(round(detect_conf, 2))
                            plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)
