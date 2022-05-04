
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

SOURCE = 'yolo Object Detection'
WEIGHTS = 'yolov5s.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False


source, weights, imgsz = SOURCE, WEIGHTS, IMG_SIZE
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
if half:
    model.half()  # to FP16
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

cap= cv2.VideoCapture(0)
prevTime = 0

prev_per = []
prev_pho = []
vector_per= []
vector_pho= []
while cap.isOpened():
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
        per_li = []
        pho_li = []
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            if names[int(cls)] == 'person':
                pe_c1, pe_c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                pe_x, pe_y = pe_c1[0] + (pe_c2[0] - pe_c1[0]) / 2, pe_c1[1] + (pe_c2[0] - pe_c1[0]) / 2
                # print("Person: {} {}".format(pe_x, pe_y))
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                per_li.append([pe_x, pe_y])
                prev_per.append(per_li)

            if names[int(cls)] == "cell phone":
                po_c1, po_c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                po_x, po_y = po_c1[0] + (po_c2[0] - po_c1[0]) / 2, po_c1[1] + (po_c2[0] - po_c1[0]) / 2
                # print("Phone: {} {}".format(po_c1[0] + (po_c2[0] - po_c1[0]) / 2, po_c1[1] + (po_c2[0] - po_c1[0]) / 2))
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                pho_li.append([po_x, po_y])
                # if prev_per:
                #     if len(prev_per) > len():
                # else:
                #     prev_pho.append(pho_li)
        print("Person", per_li)
        print("phone", pho_li)
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / (sec)
    str = "FPS : %0.1f" % fps
    cv2.putText(img0, str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.imshow(source, img0)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break




