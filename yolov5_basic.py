
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

from check import *

bc = bcolors
print(bc.OKGREEN + MESSAGE + bc.ENDC)

cap= cv2.VideoCapture(VIEDO)
prevTime = 0
while cap.isOpened():
    check = Check()
    curTime = time.time()
    ret,img0 = cap.read()
    #img0 = imutils.resize(img0, width=400)
    img = letterbox(img0 , imgsz, stride=stride)[0]


    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # print(img)
    img = torch.from_numpy(img).to(device)
    #print(img)
    img = img.half() if half else img.float()  # uint8 to fp16/32

    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    #print(img.shape)
    pred = model(img, augment=AUGMENT)[0]
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)
    #print(pred)
    det = pred[0]
    #print(det)
    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string

    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
        for *xyxy, conf, cls in reversed(det):
            draw_color = colors[int(cls)]
            detect_name = names[int(cls)]
            # detect_conf = float(conf)
            # label = detect_name
            # plot_one_box(xyxy, img0, label=label, color=draw_color, line_thickness=3)
            check.update(img0, xyxy, detect_name, draw_color)
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)
    str = "FPS : %0.1f" % fps
    cv2.putText(img0, str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.imshow(source, img0)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
