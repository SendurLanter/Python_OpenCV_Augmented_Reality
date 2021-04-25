import requests
import cv2
import numpy as np
import math
import json
import os
from objloader_simple import *
from threading import Thread
from pathlib import Path
from time import time
#MIN_MATCHES = 70
#rectangle=True

#This functions loads the target surface image
def main():
    '''homography = None
    # matrix of camera parameters (made up but works quite well for me)
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create ORB keypoint detector
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, 'reference/Untitled6.jpg'), 0)
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/spider.obj'), swapyz=True)'''  
    # init video capture
    cap = cv2.VideoCapture(0)

    while True:
        # read the current frame
        #frame=cv2.imread('test.jpg')
        ret, frame = cap.read()
        cv2.imwrite('local.jpg',frame)
        print('Transmitted task:', Path('local.jpg').stat().st_size/1024, 'KB')
        frame=open('local.jpg','rb')
        files={'file':('AR',frame,'image/jpg')}
        try:
            r=requests.post('http://34.80.232.139:80',files=files, timeout=0.1)
            with open('display.jpg','wb') as f:
                f.write(r.content)
            cv2.imshow('frame', cv2.imread('display.jpg'))
            print('Received execution result')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass

        '''# find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # match frame descriptors with model descriptors
        matches = bf.match(des_model, des_frame)
        
        # sort them in the order of their distance the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)
        #print(len(matches))
        # compute Homography if enough matches are found
        if len(matches) > MIN_MATCHES:
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            print(src_pts, dst_pts)
            start=time()
            # compute Homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if time()-start>0.02:

                if rectangle:
                    # Draw a rectangle that marks the found model in the frame
                    h, w = model.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    # project corners into frame
                    dst = cv2.perspectiveTransform(pts, homography)
                    # connect them with lines  
                    frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  

                # if a valid homography matrix was found render cube on model plane
                if homography is not None:
                    try:
                        # obtain 3D projection matrix from homography matrix and camera parameters
                        projection = projection_matrix(camera_parameters, homography)  

                        # project cube or model
                        frame = render(frame, obj, projection, model, False)

                    except:
                        pass

        # show result
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

#Render a loaded obj model into the current video frame
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

#From the camera calibration matrix and the estimated homography compute the 3D projection matrix
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
    return np.dot(camera_parameters, projection)'''

if __name__ == '__main__':
    main()