import cv2
import argparse
import dlib
from imutils import face_utils
from daugman import find_iris
import imutils
import cv2
import numpy as np
import glob
import pickle
from sklearn.externals import joblib
from keras.applications.vgg16 import preprocess_input
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.preprocessing import image


import keras

from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16

from keras.models import Model
from keras.layers import Dense, Flatten,Dropout
from keras.models import Sequential
from keras import optimizers
from keras.applications.densenet import DenseNet201


#original_model = VGG16(include_top=False)
original_model = DenseNet201(include_top=False)
#original_model = InceptionV3(include_top=False)
#original_model = InceptionResNetV2(include_top=False)
#original_model = Xception(include_top=False)

bottleneck_input  = original_model.get_layer(index=0).input
bottleneck_output = original_model.get_layer(index=-59).output
bottleneck_model  = Model(inputs=bottleneck_input,  outputs=bottleneck_output)

bottleneck_model.summary()

model = Sequential()
model.add(Flatten(input_shape=((6,6,32))))#4, 6, 512
#model.add(Dense(500, activation='relu'))
#model.add(Dropout(0.8))
model.add(Dense(2, activation='softmax'))
sgd = optimizers.SGD(lr=0.1)
#adam = optimizers.Adam(learning_rate=0.001)


model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
model.summary()
model.load_weights("iris.hdf5")


net = cv2.dnn_DetectionModel('face-detection-retail-0004.xml','face-detection-retail-0004.bin')
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor")
#args = vars(ap.parse_args())
cap = cv2.VideoCapture("video3.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("results.avi", fourcc, int(fps), (width, height))
k = 0
j = 0
detector = dlib.get_frontal_face_detector()
while(cap.isOpened):
  ret, frame = cap.read()
  if not ret:
    continue
  
  if k>60 and k%20==0:
    j = j+1
    
    blob = cv2.dnn.blobFromImage(frame, size=(300,300), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()
    for i in range(0,out.shape[2]):
        confidence = out[0,0,i,2]
        if confidence>0.5:
            xmin = int(out[0,0,i,3] * frame.shape[1])
            ymin = int(out[0,0,i,4] * frame.shape[0])
            xmax = int(out[0,0,i,5] * frame.shape[1])
            ymax = int(out[0,0,i,6] * frame.shape[0])
            face = frame[ymin:ymax,xmin:xmax]
            #face1 = imutils.resize(eye, width=300)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
            eyes = eye_cascade.detectMultiScale(face)
            
        
            for (ex, ey, ew, eh) in eyes:
              #cv2.rectangle(frame, (ex+x, ey+y), (x+ex + ew, y+ey + eh), (255, 0, 0), 2)
              #eye = face[ey:ey+eh, ex:ex+ew]
              #eye = imutils.resize(eye, width=300)
              
              if 90000>ew*eh>60000:
                cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                eye_color = face[ey:ey+eh, ex:ex+ew]
                print("======================")
              #cv2.imwrite("Train/110/110_1_{}.jpg".format(j),eye_color)
              #j = j+1
                img = cv2.resize(eye_color,(300,300))
                img = img[40:270, 40:270]
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                minimal_iris_radius = 60
                answer = find_iris(gray_img, minimal_iris_radius)
                print("answer:{}".format(answer))
                iris_center, iris_rad = answer
                out_iris = img.copy()
                new_roi = out_iris[iris_center[1]-iris_rad:iris_center[1]+iris_rad, iris_center[0]-iris_rad:iris_center[0]+iris_rad]
                new_roi = cv2.resize(new_roi,(200,200))
                new_roi = new_roi/255
                new_roi = new_roi.reshape(1,200,200,3)
                test_shape = bottleneck_model.predict(new_roi).shape

                shape = (1,test_shape[1],test_shape[2],test_shape[3])
                bottelneck_features = []
                bottelneck_features.append(bottleneck_model.predict(new_roi))
                bottelneck_features=np.array(bottelneck_features)
                bottelneck_features =  bottelneck_features.reshape(shape)
                result = model.predict(bottelneck_features)
                idx = np.argmax(result[0])
                if idx==0:
                    cv2.putText(frame,"Doanh", (xmin+10,ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame,"Xuan", (xmin+10,ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,0), 2, cv2.LINE_AA)
                #new_roi = out_iris[iris_center[1]-iris_rad:iris_center[1]+iris_rad, iris_center[0]-iris_rad:iris_center[0]+iris_rad]
                #new_roi = cv2.resize(new_roi,(200,200))
                #cv2.circle(out_iris, iris_center, iris_rad, (0, 0, 255), 1)
                cv2.circle(face, (int((iris_center[0]+40)*eh/300)+ex,int((iris_center[1]+40)*eh/300)+ey), iris_rad, (0, 0, 255), 1)
                cv2.circle(frame, (xmin+int((iris_center[0]+40)*eh/300)+ex,ymin+int((iris_center[1]+40)*eh/300)+ey), iris_rad, (0, 0, 255), 1)
            try:
              face = cv2.resize(face, (600,int(600*(ymax-ymin)/(xmax-xmin))))
              #cv2.imshow("test",face)
            except Exception as e:
              print(str(e))
            key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
            if key == ord("q"):
              break
  writer.write(frame)
              
  k = k + 20
  
cap.realease()
cv2.destroyAllWindows()
