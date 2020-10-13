import numpy as np
import argparse
import pickle
import socket
import cv2
import os
from objloader_simple import *
from threading import Thread
from time import time

def projection_matrix(camera_parameters, homography):
	# Compute rotation along the x and y axis as well as the translation
	homography = homography * (-1)
	rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
	col_1 = rot_and_transl[:, 0]
	col_2 = rot_and_transl[:, 1]
	col_3 = rot_and_transl[:, 2]
	# normalise vectors
	l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
	rot_1 = col_1 / l
	rot_2 = col_2 / l
	translation = col_3 / l
	# compute the orthonormal basis
	c = rot_1 + rot_2
	p = np.cross(rot_1, rot_2)
	d = np.cross(c, p)
	rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_3 = np.cross(rot_1, rot_2)
	# finally, compute the 3D projection matrix from the model to the current frame
	projection = np.stack((rot_1, rot_2, rot_3, translation)).T
	return np.dot(camera_parameters, projection)

def render(img, obj, projection, model, color=False):
	vertices = obj.vertices
	scale_matrix = np.eye(3)*1
	h, w = model.shape

	for face in obj.faces:
		face_vertices = face[0]
		points = np.array([vertices[vertex - 1] for vertex in face_vertices])
		points = np.dot(points, scale_matrix)
		# render model in the middle of the reference surface. To do so, model points must be displaced
		points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
		dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
		imgpts = np.int32(dst)
		cv2.polylines(img, imgpts, isClosed=True, color=(127,255,0), thickness=3)
		cv2.fillConvexPoly(img, imgpts,(0,0,0))
	return img


parser = argparse.ArgumentParser(description='Augmented reality application')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
args = parser.parse_args()

homography = None
camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
dir_name = os.getcwd()
model = cv2.imread(os.path.join(dir_name, 'reference/model.jpg'), 0)
kp_model, des_model = orb.detectAndCompute(model, None)
obj = OBJ(os.path.join(dir_name, 'models/Only_Spider_with_Animations_Export.obj'), swapyz=True)  

cap = cv2.VideoCapture(0)

HOST, PORT = "", 11111
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
while 1:
	s.listen(0)
	client, address = s.accept()
	print(str(address)+" connected")

	while 1:
		try:
			frame = pickle.loads(client.recv(3000000))
			kp_frame, des_frame = orb.detectAndCompute(frame, None)
			matches = bf.match(des_model, des_frame)
			matches = sorted(matches, key=lambda x: x.distance)
			
			src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
			dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
			homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			if homography is not None:
				try:
					projection = projection_matrix(camera_parameters, homography)  
					frame = render(frame, obj, projection, model, False)
				except:
					pass
			frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)

			cv2.imshow('frame', frame)
		except:
			data = client.recv(6000000)
			print(":(")
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	client.close()