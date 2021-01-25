# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np

eye_left = [37, 38, 39, 40, 41, 42]
eye_right = [43, 44, 45, 46, 47, 48]
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


j=0
# loop over the frames from the video stream
while True:
  # grab the frame from the video stream, resize it to have a
  # maximum width of 400 pixels, and convert it to grayscale
  frame = vs.read()
  frame = imutils.resize(frame, width=400)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # detect faces in the grayscale frame
  rects = detector(gray, 0)
	
	# loop over the face detections
  
  for rect in rects:
    left_eye_coor_x = []
    left_eye_coor_y = []
    right_eye_coor_x = []
    right_eye_coor_y = []
	  # convert the dlib rectangle into an OpenCV bounding box and
		# draw a bounding box surrounding the face
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	  # use our custom dlib shape predictor to predict the location
	  # of our landmark coordinates, then convert the prediction to
	  # an easily parsable NumPy array
    shape = predictor(gray, rect)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    for num in range(shape.num_parts):
      
      if num+37 in eye_left:
	      print("OK!")
	      left_eye_coor_x.append(shape.parts()[num].x)
	      left_eye_coor_y.append(shape.parts()[num].y)
      else:
	      right_eye_coor_x.append(shape.parts()[num].x)
	      right_eye_coor_y.append(shape.parts()[num].y)
    try:
      x_left_min = min(left_eye_coor_x)
      x_left_max = max(left_eye_coor_x)
      y_left_min = min(left_eye_coor_y)
      y_left_max = max(left_eye_coor_y)
	
      left_eye = frame[y_left_min:y_left_max, x_left_min:x_left_max]
      left_eye = cv2.cvtColor(left_eye,cv2.COLOR_BGR2GRAY)
      left_eye = imutils.resize(left_eye, width=320)
      cv2.imwrite("./custom_data/001_1left_{}.jpg".format(j),left_eye)

      x_right_min = min(right_eye_coor_x)
      x_right_max = max(right_eye_coor_x)
      y_right_min = min(right_eye_coor_y)
      y_right_max = max(right_eye_coor_y)

      right_eye = frame[y_right_min:y_right_max, x_right_min:x_right_max]
      right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
      right_eye = imutils.resize(right_eye, width=320)
      cv2.imwrite("./custom_data/001_1right_{}.jpg".format(j),right_eye)
      
      j = j+1
    except:
      print("No eyes detected")
    shape = face_utils.shape_to_np(shape)
	
	  # loop over the (x, y)-coordinates from our dlib shape
	  # predictor model draw them on the image
    for (sX, sY) in shape:
	    cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)
	
	
    # show the frame
  cv2.imshow("Frame", frame)
  cv2.imwrite("eye.jpg",frame)
  key = cv2.waitKey(1) & 0xFF
  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
	  break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
