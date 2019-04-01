import cv2
import sys
a = cv2.imread(sys.argv[1])
cv2.imwrite(sys.argv[1][:-4]+"_c.jpg", a)
