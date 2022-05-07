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


#### TODO ####
"""
---> Person.detect_person()
1. 사람이 탐지 되었을 경우, 이륜차 탐지
2. 사람의 벡터값과 이륜차의 벡터값을 비교
3. 사람의 벡터값과 이륜차의 벡터값이 유사할 경우, 헬멧 착용여부 확인
"""

prev_pho = []
vector_pho = []
class Bike:
    def __init__(self):
        self.state = "detect"
        self.pho_li = []

    def track_bike(self):
        global prev_pho, vector_pho
        if prev_pho != []:
            for pho in self.pho_li:
                for prev in prev_pho:
                    if TRACKING_SPEED[0] < abs(prev[0] - pho[0]) < TRACKING_SPEED[1] and TRACKING_SPEED[0] < abs(prev[1] - pho[1]) < TRACKING_SPEED[1]:
                        vector_pho.append([prev[0] - pho[0], prev[1] - pho[1]])
        prev_pho = self.pho_li

    def check_bike(self, vector_per):
        if len(vector_per) > 0 and len(vector_pho)> 0:
            print("=============================")
            print("Per_Vector: ", vector_per)
            print("Bike_Vec", vector_pho)

    def detect_bike(self, img0, xyxy, detect_name, draw_color, detect_conf):
        global prev_pho, vector_pho
        label = detect_name +" "+ str(round(detect_conf, 2))
        if detect_name == 'cell phone':
            pe_c1, pe_c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            pe_x, pe_y = pe_c1[0] + (pe_c2[0] - pe_c1[0]) / 2, pe_c1[1] + (pe_c2[0] - pe_c1[0]) / 2
            plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)
            self.pho_li.append([pe_x, pe_y])
            # print("cell Phone: ", self.pho_li)
            # print("Prev: ", prev_pho)
            self.track_bike()