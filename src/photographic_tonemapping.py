import numpy as np
import cv2
import sys
import copy
import math

def calculate_average_and_max(lumi):
	total=0
	max_lumi=-1
	for i in range(lumi.shape[0]):
		for o in range(lumi.shape[1]):
			if(lumi.item(i,o)>max_lumi):
				max_lumi=lumi.item(i,o)
			total+=math.log(lumi.item(i,o)+10**(-10))

	
	return (math.exp(total / (lumi.shape[0]*lumi.shape[1]))) , max_lumi

def make_luminence(hdr):
	lumi=np.zeros((hdr.shape[0],hdr.shape[1]),np.float32)
	for i in range(lumi.shape[0]):
		for o in range(lumi.shape[1]):
			lumi.itemset(i,o,(0.27*(hdr.item(i,o,0))+0.67*(hdr.item(i,o,1))+0.06*(hdr.item(i,o,2))))
	return lumi

def global_operator(lumi,key,average,max_lumi):
	max_lumi=(float(key)/average)*max_lumi
	for i in range(lumi.shape[0]):
		for o in range(lumi.shape[1]):
			lumi.itemset(i,o,(float(key)/(average))*lumi.item(i,o))
	print("max_lumi",max_lumi)
	for i in range(lumi.shape[0]):
		for o in range(lumi.shape[1]):
			temp=lumi.item(i,o)
			lumi.itemset(i,o,temp*((1+temp/(max_lumi*max_lumi))/(1+temp)))
			print("temp",temp,"modified",lumi.item(i,o))

def bivariate_gaussian(x,y,sharp):
	return (1/(math.pi*sharp*sharp))*math.exp(-(x*x+y*y)/(sharp*sharp))

def local_dodging_and_burning(lumi,key,average,max_lumi,sharp):
	#huge sharp would result in an image with huge contrast. Default to 3.
	max_lumi=5*(float(key)/average)*max_lumi
	for i in range(lumi.shape[0]):
		for o in range(lumi.shape[1]):
			lumi.itemset(i,o,(float(key)/(average))*lumi.item(i,o))
	lumi_copy=copy.deepcopy(lumi)
	print(lumi)
	for i in range(lumi.shape[0]):
		for o in range(lumi.shape[1]):
			temp=lumi.item(i,o)
#			deno=0
#			numi=0
#			for m in range(-5,6):
#				for n in range(-5,6):
#					if i+m < 0 or i+m >= lumi.shape[0] or o+n < 0 or o+n >= lumi.shape[1]:
#						continue
#					
#					norm=bivariate_gaussian(m,n,sharp)
#					deno+=norm
#					numi+=norm*lumi_copy.item(i+m,o+n)
#			
#			lumi.itemset(i,o,temp*((1+temp/(max_lumi*max_lumi))/(1+(numi/deno))))
			lumi.itemset(i,o,temp*((1+temp/(max_lumi*max_lumi))/(1+temp)))

def produce_new_image(lumi,lumi_original,hdr,image_name):
	tone_mapped=np.zeros((hdr.shape[0],hdr.shape[1],3),np.uint8)
	for i in range(hdr.shape[0]):
		for o in range(hdr.shape[1]):
			for z in range(3):
				V=hdr.item(i,o,z)*lumi.item(i,o)*255/(lumi_original.item(i,o)+10**(-10))
				V=round(V)
				#print(i,o,z,V,end="")
				if V >= 255:
					V=255
				tone_mapped.itemset((i,o,z),V)
				#print(tone_mapped.item(i,o,z))	
	image_name+="_tone_mapped.png"
	cv2.imwrite(image_name,tone_mapped)

#key is provided for the overall luminence of the image.Darker image should have a key with low value. Brighter image should have higher key.
#higher sharp incease the sharpness or contrast of a image.
def photographic_tonemapping(image_name,key,sharp): 
	sharp=float(sharp)
	key=float(key)
	hdr=cv2.imread(image_name,cv2.IMREAD_ANYDEPTH)
	print(hdr.dtype,hdr.shape)
	#for color image only
	if (hdr.dtype != np.float32) and (hdr.shape[2] == 3):
		print("image should have be in np.float32 format.\
			And it should have three channels")
		exit(-1)
	
	#build a luminence image based on colored hdr
	lumi=make_luminence(hdr)
	lumi_original=copy.deepcopy(lumi)
	
	average, max_lumi  =calculate_average_and_max(lumi)
	print(max_lumi)
	#global_operator(lumi,key,average,max_lumi)

	local_dodging_and_burning(lumi,key,average,max_lumi,sharp)
	
	ldr_name = image_name

	produce_new_image(lumi,lumi_original,hdr,image_name)

	tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)

	ldrDrago = tonemapDrago.process(hdr)
	ldrDrago = 3 * ldrDrago
	cv2.imwrite(ldr_name[:-4]+"_cv2.jpg", ldrDrago * 255)

if __name__=="__main__":
	
	if len(sys.argv) != 4:
		print("usage: photographic_tomemapping <hdr_image> <key> <sharp=3>")
	photographic_tonemapping(sys.argv[1],sys.argv[2],sys.argv[3])





