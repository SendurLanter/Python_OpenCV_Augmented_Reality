from time import time
import cv2
cap = cv2.VideoCapture(0)
while 1:
	start=time()
	ret, frame = cap.read()
	cv2.imshow('frame', frame)
	print(time()-start)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break