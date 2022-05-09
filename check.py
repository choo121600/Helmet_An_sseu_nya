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
from detects.bike import Bike
from detects.helmet import Helmet
from detects.person import Person

class Check:
    def __init__(self):
        self.helmet = Helmet()
        self.bike = Bike()
        self.person = Person()

    def check_helmet(self, img, bike_state, bike_x_li, bike_y_li, helmet_li, sec):
        global IMG_CNT
        if bike_state == "Move":
            if len(helmet_li) > 0:
                for bike_x in bike_x_li:
                    for helmet in helmet_li:
                        if bike_x[0] < helmet[0] < bike_x[1]:
                            print("Pass")
                            break
                    if round(sec) % 2 == 0:
                        IMG_CNT += 1
                        print(IMG_CNT)
                        cv2.imwrite("./img/helmet{}.jpg".format(IMG_CNT), img)
                        print("No helmet")
            else:
                if round(sec) % 2 == 0:
                    IMG_CNT += 1
                    print()
                    cv2.imwrite("./img/helmet{}.jpg".format(IMG_CNT), img)
                    print("No helmet total: {}".format(IMG_CNT))

    def update(self, img0, xyxy, detect_name, draw_color, sec):
        self.bike.detect_bike(img0, xyxy, detect_name, draw_color)
        self.person.detect_person(img0, xyxy, detect_name, draw_color)
        self.helmet.detect_helmet(img0, xyxy, detect_name, draw_color)
        self.check_helmet(img0, self.bike.bike_state, self.bike.bike_x_li, self.bike.bike_y_li, self.helmet.helmet_li, sec)