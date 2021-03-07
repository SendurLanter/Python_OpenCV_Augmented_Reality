from threading import Thread
from time import time,sleep
import requests
import socket
import cv2

#cap=cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://192.168.8.107:4747/mjpegfeed')
s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def sndv():
	while cap.isOpened():
		ret, frame = cap.read()
		cv2.imwrite('client_buffer.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 25])
		with open('client_buffer.jpg','rb') as f:
			s.sendto(f.read(), ("140.112.20.181", 12345))
		start=time()
		try:
			s.settimeout(0.15)
			data = s.recv(3000000)
			with open('client_save.jpg','wb') as f:
				f.write(data)
			frame=cv2.imread('client_save.jpg')
			cv2.imshow('frame', frame)
		except:
			print(":(")
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		print('FPS: ', 1/(time()-start) )

Thread(target = sndv).start()