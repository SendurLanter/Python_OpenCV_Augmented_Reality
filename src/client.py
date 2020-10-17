from threading import Thread
from time import time,sleep
import requests
import socket
import cv2

cap=cv2.VideoCapture(0)
tx=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tx.connect(("140.112.20.181", 11111))
rx=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
rx.connect(("140.112.20.181", 22222))
print("connected")

def sndv():
	while cap.isOpened():
		ret, frame = cap.read()
		cv2.imwrite('client_buffer.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 25])
		with open('client_buffer.jpg','rb') as f:
			tx.sendall(f.read())
def rcvv():
	while cap.isOpened():
		start=time()
		try:
			data = rx.recv(3000000)
			with open('client_save.jpg','wb') as f:
				f.write(data)
			frame=cv2.imread('client_save.jpg')
			cv2.imshow('frame', frame)
		except:
			data = rx.recv(6000000)
			print(":(")
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		print('FPS: ', 1/(time()-start) )

Thread(target = sndv).start() 
Thread(target = rcvv).start()