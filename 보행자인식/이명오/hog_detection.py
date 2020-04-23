

#-*- coding:utf-8 -*-

from __future__ import print_function
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
import skimage
from skimage.feature import hog

'''
hog+SVM을 이용하여 pedestrian detection을 수행하고 
NMS 알고리즘을 통해 Threshold에 따른 bounding box를 걸러냅니다.

HOGDescriptor, SVM은 opencv를 이용하였고 
NMS는 imutils을 이용하였습니다.

또한 HOG feature가 어떻게 뽑히는지 시각화하기 위한 결과를 추가하였으며 해당 작업은
opencv로 해보려고 했는데 잘 되지 않아서 skimage를 이용하였습니다.

'''


# 이미지 로드 및 복사
image = cv2.imread('test.png')
image2 = image.copy()


# opencv의 HOGDescriptor + setSVMDetector를 이용하여 HOG feature 추출 및 보행자를 검출합니다.
hog_opencv = cv2.HOGDescriptor()
hog_opencv.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# window stirde, padding, scale 등의 파라미터 값을 설정하고 바운딩 박스 좌표를 리턴
rects, weight = hog_opencv.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.5)

# 검출된 영역을 빨간색 바운딩 박스로 이미지에 그림
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

# NMS 알고리즘을 사용하여 중복된 box를 걸러냄
rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)


# 중복된 box를 걸러낸 후 남은 box를 초록색 박스로  그림
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(image2, (xA, yA), (xB, yB), (0, 255, 0), 2)


print("{} original boxes, {} after suppression".format(len(rects), len(pick)))

# 결과 확인
cv2.imshow("Before NMS", image)
cv2.imshow("After NMS", image2)



# 입력 이미지에 따라 HOG feature결과가 어떻게 추출되는 skimage 라이브러리를 이용하여 시각화
fd, hog_img = hog(cv2.imread('test.png'), 9, (8, 8), (1, 1), visualize=True)
hog_img = skimage.exposure.rescale_intensity(hog_img, (0, 10))

cv2.imshow("Hog feature", hog_img)
cv2.waitKey(0)
