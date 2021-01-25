import cv2
import os
import numpy as np

images = os.listdir("Viet")
i = 0
for image in images:
  img = cv2.imread("Viet/"+image)
  img = cv2.resize(img,(280,280))
  img_copy = np.zeros((280,320,3), np.uint8)
  img_copy[0:280,20:300] = img
  cv2.imwrite("Viet/108_1_{}.jpg".format(i),img_copy)
  i = i+1
  print(img.shape)
  

