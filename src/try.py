import mtb
import numpy as np
import cv2
x=np.zeros((100,100,3))
y=np.zeros((100,100,3))
for i in range(100):
	for o in range(100):
		if o < 80 and o > 60:
			x.itemset(i,o,0,200)
			x.itemset(i,o,1,200)
			x.itemset(i,o,2,200)
		if o < 70 and o > 50:
			y.itemset(i,o,0,200)
			y.itemset(i,o,1,200)
			y.itemset(i,o,2,200)
cv2.imwrite("x.png",x)
cv2.imwrite("y.png",y)

