from flask import Flask, request
import cv2

headers = {'Content-Type': 'application/json'}
app = Flask(__name__)
app.debug = True
@app.route('/',methods=['POST'])
def sockeeet():
	req = request.get_json()
	print(req)
	value[req['instance']]=req['data']


def projection_matrix(camera_parameters, homography):
	homography = homography * (-1)
	rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
	col_1 = rot_and_transl[:, 0]
	col_2 = rot_and_transl[:, 1]
	col_3 = rot_and_transl[:, 2]
	l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
	rot_1 = col_1 / l
	rot_2 = col_2 / l
	translation = col_3 / l
	c = rot_1 + rot_2
	p = np.cross(rot_1, rot_2)
	d = np.cross(c, p)
	rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
	rot_3 = np.cross(rot_1, rot_2)
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
		points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
		dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
		imgpts = np.int32(dst)
		cv2.polylines(img, imgpts, isClosed=True, color=(127,255,0), thickness=3)
		cv2.fillConvexPoly(img, imgpts,(0,0,0))
	return img

def main():
	homography = None
	camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
	orb = cv2.ORB_create()
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	dir_name = os.getcwd()
	model = cv2.imread(os.path.join(dir_name, 'reference/model.jpg'), 0)
	kp_model, des_model = orb.detectAndCompute(model, None)
	obj = OBJ(os.path.join(dir_name, 'models/Only_Spider_with_Animations_Export.obj'), swapyz=True)  

	cap = cv2.VideoCapture(0)

	HOST, PORT = "", 12345
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind((HOST, PORT))
	while 1:
		s.listen(0)
		client, address = s.accept()
		print(str(address)+" connected")
		while 1:
			try:
				frame = pickle.loads(client.recv(3000000))
				print('0')
				kp_frame, des_frame = orb.detectAndCompute(frame, None)
				matches = bf.match(des_model, des_frame)
				matches = sorted(matches, key=lambda x: x.distance)
				print('1')
				if len(matches) > MIN_MATCHES:
					src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
					dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
					homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
					if args.rectangle:
						h, w = model.shape
						pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
						dst = cv2.perspectiveTransform(pts, homography)
						frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
					if homography is not None:
						try:
							projection = projection_matrix(camera_parameters, homography)  
							frame = render(frame, obj, projection, model, False)
						except:
							pass
				print('2')
				#client.send(pickle.dumps(frame))
			except:
				data = client.recv(10000000)

		client.close()