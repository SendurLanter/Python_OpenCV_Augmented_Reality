from time import time
from PIL import Image

while 1:
	start=time()
	frame=Image.open('profile.png')
	frame.show()
	print(time()-start)