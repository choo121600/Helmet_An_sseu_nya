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
from settings import *
from Bike.bike import *

prev_per = []

class Person:
    def __init__(self):
        self.state = "detect"
        self.per_li = []
        self.bike = Bike()
        self.vector_per = []

    def track_person(self):
        global prev_per
        if prev_per != []:
            for per in self.per_li:
                for prev in prev_per:
                    if TRACKING_SPEED[0] < abs(prev[0] - per[0]) < TRACKING_SPEED[1] and TRACKING_SPEED[0] < abs(prev[1] - per[1]) < TRACKING_SPEED[1]:
                        self.vector_per.append([prev[0] - per[0], prev[1] - per[1]])
        prev_per = self.per_li

    def check_person(self, vector_per):
        self.bike.check_bike(vector_per)

    def detect_person(self, img0, xyxy, detect_name, draw_color, detect_conf):
        global prev_per
        label = detect_name +" "+ str(round(detect_conf, 2))
        if detect_name == 'person':
            pe_c1, pe_c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            pe_x, pe_y = pe_c1[0] + (pe_c2[0] - pe_c1[0]) / 2, pe_c1[1] + (pe_c2[0] - pe_c1[0]) / 2
            plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)
            self.per_li.append([pe_x, pe_y])
            # print("Person: ", self.per_li)
            # print("Prev: ", prev_per)
            self.track_person()
        self.bike.detect_bike(img0, xyxy, detect_name, draw_color, detect_conf)
        if self.vector_per:
            self.check_person(self.vector_per)
