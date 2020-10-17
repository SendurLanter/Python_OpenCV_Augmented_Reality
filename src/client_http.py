from flask import Flask, request
import cv2

cap = cv2.VideoCapture(0)

headers = {'Content-Type': 'application/json'}
app = Flask(__name__)
app.debug = True
@app.route('/',methods=['POST'])
def sockeeet():