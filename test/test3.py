from time import time
import cv2

while 1:
	start=time()
	frame=cv2.imread('system.png')
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	print(time()-start)