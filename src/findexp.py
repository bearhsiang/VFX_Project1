from exif import Image
import sys


if len(sys.argv) != 3:
	print("usage: python3 findexp.py [src_dir] [img_list]")
	print("e.g: python3 findexp.py ./myimg/ myimg_list.txt")
	exit()
d = sys.argv[1]
if d[-1] != '/':
	d = d+'/'

for i in open(sys.argv[2], 'r'):
	with open(d+i[:-1], 'rb') as image_file:
		img = Image(image_file)
		print(i[:-1], img.exposure_time)