# Object Detection

import cv2

thres = 0.5 # Threshold to detect object
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classFile = 'coco.names'

# import names.txt file
f = open(classFile, 'r')
classNames = f.read().rstrip('\n').split('\n')

while True:
    success, img = cap.read()
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 0, 0), thickness=2)
            cv2.putText(img, classNames[classId-1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 450, box[1] + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

    cv2.imshow('Output', img)
    cv2.waitKey(1)
