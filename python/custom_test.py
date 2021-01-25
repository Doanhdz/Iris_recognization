##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import argparse
from time import time

from fnc.extractFeature_custom import extractFeature
from fnc.matching import matching
import cv2
import argparse
import dlib
from imutils import face_utils
import time
import numpy as np
# 109-Doanh
#------------------------------------------------------------------------------
#	Argument parsing
#------------------------------------------------------------------------------

eye_left = [37, 38, 39, 40, 41, 42]
eye_right = [43, 44, 45, 46, 47, 48]
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
net = cv2.dnn_DetectionModel('../face-detection-retail-0004.xml','../face-detection-retail-0004.bin')
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
parser = argparse.ArgumentParser()

parser.add_argument("--temp_dir", type=str, default="./templates/temp/",
					help="Path to the directory containing templates.")

parser.add_argument("--thres", type=float, default=0.38,
					help="Threshold for matching.")
args = parser.parse_args()


# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("eye_landmarks.dat")
# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
#cap = cv2.VideoCapture('rtsp://admin:CIST2020@192.168.1.248:554')
cap = cv2.VideoCapture('video3.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("results.avi", fourcc, int(fps), (width, height))

# loop over the frames from the video stream
while cap.isOpened():
	# grab the frame from the video stream, resize it to have a
	# maximum width of 400 pixels, and convert it to grayscale
  ret, frame = cap.read()
  if not ret:
    continue
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        eyes = eye_cascade.detectMultiScale(face)
        print("++++++++++++++++")
        eye_data = []
        for (ex, ey, ew, eh) in eyes:
          
          #eye = face[ey:ey+eh, ex:ex+ew]
          #eye = imutils.resize(eye, width=300)
          eye_color = face[ey:ey+eh, ex:ex+ew]
          if eh*ew > 70000:
            print(eh,ew)
            cv2.rectangle(frame, (ex+xmin, ey+ymin), (xmin+ex + ew, ymin+ey + eh), (255, 0, 0), 2)
            img = cv2.resize(eye_color,(280,280))
            img_copy = np.zeros((280,320,3), np.uint8)
            img_copy[0:280,20:300] = img
            print('>>> Start verifying {}\n'.format(img_copy))
            template, mask= extractFeature(img_copy)
            result = matching(template, mask, args.temp_dir, args.thres)
            if result == -1:
              print('>>> No registered sample.')
              cv2.putText(frame,"Unknown", (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,0), 2, cv2.LINE_AA)

            elif result == 0:
              print('>>> No sample matched.')
              cv2.putText(frame,"Unknown", (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,0), 2, cv2.LINE_AA)

            else:
              for res in result:
                idx = res.split(".")[0].split("_")[0]
                if idx == "109":
                  cv2.putText(frame,"Doanh", (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,0), 2, cv2.LINE_AA)
                elif idx == "110": 
                  cv2.putText(frame,"Xuan",(xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,0), 2, cv2.LINE_AA)
                else:
                  cv2.putText(frame,"Unknown", (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,0), 2, cv2.LINE_AA)
                break
          # print('>>> {} samples matched (descending reliability):'.format(len(result)))
          #     cv2.putText(frame,"Match", (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,0), 2, cv2.LINE_AA)


        # #face = frame[ymin:ymax,xmin:xmax]
        # rect = dlib.rectangle(int(xmin),int(ymin),int(xmax),int(ymax))
        # left_eye_coor_x = []
        # left_eye_coor_y = []
        # right_eye_coor_x = []
        # right_eye_coor_y = []
        # # convert the dlib rectangle into an OpenCV bounding box and
        # # draw a bounding box surrounding the face
        # # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # # use our custom dlib shape predictor to predict the location
        # # of our landmark coordinates, then convert the prediction to
        # # an easily parsable NumPy array
        # #shape = predictor(gray, rect)
        # shape = predictor(gray, rect)
        # #shape_1 = face_utils.shape_to_np(shape)
        # #for (x,y) in shape_1:
        # #  cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)
        # landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        # for num in range(shape.num_parts):
          
        #   if num+37 in eye_left:
        #     left_eye_coor_x.append(shape.parts()[num].x)
        #     left_eye_coor_y.append(shape.parts()[num].y)
        #   else:
        #     right_eye_coor_x.append(shape.parts()[num].x)
        #     right_eye_coor_y.append(shape.parts()[num].y)
        # if left_eye_coor_x:
          
        #   x_left_min = min(left_eye_coor_x)-10
        #   x_left_max = max(left_eye_coor_x)+10
        #   y_left_min = min(left_eye_coor_y)-10
        #   y_left_max = max(left_eye_coor_y)+10
        #   cv2.rectangle(frame, (x_left_min, y_left_min), (x_left_max, y_left_max), (0, 255, 0), 2)

        #   left_eye = frame[y_left_min:y_left_max, x_left_min:x_left_max]
        #   img = cv2.resize(left_eye,(280,280))
        #   img_copy = np.zeros((280,320,3), np.uint8)
        #   img_copy[0:280,20:300] = img
        #   #left_eye = cv2.cvtColor(left_eye,cv2.COLOR_BGR2GRAY)
        #   #left_eye = imutils.resize(left_eye, width=400)
        #   #iris_segment.process_image(left_eye)

        #   x_right_min = min(right_eye_coor_x)-10
        #   x_right_max = max(right_eye_coor_x)+10
        #   y_right_min = min(right_eye_coor_y)-10
        #   y_right_max = max(right_eye_coor_y)+10
        #   cv2.rectangle(frame, (x_right_min, y_right_min), (x_right_max, y_right_max), (0, 255, 0), 2)

        #   right_eye = frame[y_right_min:y_right_max, x_right_min:x_right_max]
        #   img1 = cv2.resize(right_eye,(280,280))
        #   img_copy1 = np.zeros((280,320,3), np.uint8)
        #   img_copy1[0:280,20:300] = img
          # print('>>> Start verifying {}\n'.format(img_copy))
          # template, mask= extractFeature(img_copy)
          # template1, mask1= extractFeature(img_copy1)


        # Matching
          # result0 = matching(template, mask, args.temp_dir, args.thres)
          # result1 = matching(template1, mask1, args.temp_dir, args.thres)


  #         if result0 == -1 and result1==-1:
  #           print('>>> No registered sample.')
  #           cv2.putText(frame,"Unknown", (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,0), 2, cv2.LINE_AA)

  #         elif result0 == 0 and result1 == 0:
  #           print('>>> No sample matched.')
  #           cv2.putText(frame,"Unknown", (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,0), 2, cv2.LINE_AA)

  #         else:
  #         #print('>>> {} samples matched (descending reliability):'.format(len(result)))
  #           cv2.putText(frame,"Match", (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,0), 2, cv2.LINE_AA)
  #           #idx_left = 0
  #           #idx_right = 0
            
  #           #for res in result0:
  #            # idx_left = res.split(".")[0].split("_")[0]
  #           #for res in result1:
  #            # idx_right = res.split(".")[0].split("_")[0]
  #             #if idx_left == "109" and idx_left==idx_right:
  #              # cv2.putText(frame,str(idx_left), (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,0), 2, cv2.LINE_AA)
  #             #elif idx_left == "110" and idx_left==idx_right: 
  #              # cv2.putText(frame,str(idx_left), (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,0), 2, cv2.LINE_AA)
  #             #else:
  #              # cv2.putText(frame,"Unknown", (xmin-10,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 5,(255,255,0), 2, cv2.LINE_AA)
  #             #break
  #         # for res in result:
  #          #  print("\t", res)
  # #cv2.imshow("frame",frame)
  writer.write(frame)
  #if cv2.waitKey(1) & 0xFF == ord('q'):
   # break
  # for rect in rects:
  #   (x, y, w, h) = face_utils.rect_to_bb(rect)
  #   face = gray[y:y + h, x:x + w]
  #   face_color = frame[y:y + h, x:x + w]
  #   eyes = eye_cascade.detectMultiScale(face)

  #   for (ex, ey, ew, eh) in eyes:
  #     eye = face_color[ey:ey+eh, ex:ex+ew]
  #     cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,255,0),1)
  #     #eye_color = face_color[ey:ey+eh, ex:ex+ew]
      

      # img = cv2.resize(left_eye,(280,280))
      # img_copy = np.zeros((280,320,3), np.uint8)
      # img_copy[0:280,20:300] = img
      # detect faces in the grayscale frame



		# show the frame
  
  
cap.realease()
writer.release()
cv2.destroyAllWindows()

