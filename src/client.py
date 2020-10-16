from time import time
import pickle
import socket
import cv2

cap = cv2.VideoCapture(0)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("140.112.20.181", 12345))
print("connected")

while 1:
	start=time()
	ret, frame = cap.read()
	s.sendall(pickle.dumps(frame))
	try:
		frame = pickle.loads(s.recv(3000000))
		cv2.imshow('frame', frame)
	except:
		data = s.recv(6000000)
		print(":(")
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	print('FPS: ', 1/(time()-start) )
s.close()