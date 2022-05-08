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

prev_helmet = []
class Helmet:
    def __init__(self):
        self.state = "Stop"
        self.helmet_li = []
    
    def detect_helmet(self, img0, xyxy, detect_name, draw_color):
        global prev_per
        if detect_name == 'Helmet':
            helmet_c1, helmet_c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            helmet_x_mid, helmet_y_mid = int((helmet_c1[0] + helmet_c2[0]) / 2), int((helmet_c1[1] + helmet_c2[1]) / 2)
            self.helmet_li.append([helmet_x_mid, helmet_y_mid])

            label = detect_name +" "+ self.state
            plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)
