import pickle
import socket
import cv2
cap = cv2.VideoCapture(0)
HOST, PORT = "127.0.0.1", 11111
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
print("connected to "+HOST)

while 1:
	ret, frame = cap.read()
	s.sendall(pickle.dumps(frame))
s.close()