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
import time # time 라이브러리

prev_pho = []

class Bike:
    def __init__(self):
        self.state = "detect"
        self.pho_li = []

    def detect_bike(self, img0, xyxy, detect_name, draw_color, detect_conf):
        global prev_pho
        label = detect_name +" "+ str(round(detect_conf, 2))
        if detect_name == 'cell phone':
            pe_c1, pe_c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            pe_x, pe_y = pe_c1[0] + (pe_c2[0] - pe_c1[0]) / 2, pe_c1[1] + (pe_c2[0] - pe_c1[0]) / 2
            plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)
            self.pho_li.append([pe_x, pe_y])
            print("cell Phone: ", self.pho_li)
            print("Prev: ", prev_pho)
            self.track_bike()
        
    def track_bike(self):
        global prev_pho
        self.vector_pho = []
        if prev_pho != []:
            for pho in self.pho_li:
                for prev in prev_pho:
                    if 5 < abs(prev[0] - pho[0]) < 50 and 5 < abs(prev[1] - pho[1]) < 50:
                        self.vector_pho.append([prev[0] - pho[0], prev[1] - pho[1]])
                        print("Vector: ", self.vector_pho)
        prev_pho = self.pho_li
