from time import time, sleep
import threading
import cv2
#cap = cv2.VideoCapture('http://192.168.8.107:4747/mjpegfeed')
class webcam:
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
		self.frame=[]
	def RSTP(self):
		while 1:
			ret, self.frame = self.cap.read()
	def start(self):
		threading.Thread(target=self.RSTP, daemon=True, args=()).start()
	def getframe(self):
		return self.frame

cam=webcam()
cam.start()
sleep(1)
count=0
start=time()
old=[]
while 1:
	frame=cam.getframe()
	if frame==old:
		count+=1
		old=frame
	#ret, frame = cap.read()
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if (time()-start)>10:
		break
print(count/10)