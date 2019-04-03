import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt



ignoreRatio = 0.1

def shift(m, n, axis):

	if n == 0:
		return m

	x = np.roll(m, n, axis=axis)
	if axis == 1:
		if n > 0:
			x[:, :n] = False
		else:
			x[:, n:] = False
	else:
		if n > 0:
			x[:n, :] = False
		else:
			x[n:, :] = False
	return x

def count_shift(im1, im2):
	print(im1, im2)
	fimgs = []
	mimgs = []
	fimg = cv2.imread(im1)
	mimg = cv2.imread(im2)

	fimg_g = cv2.cvtColor(fimg, cv2.COLOR_BGR2GRAY)
	mimg_g = cv2.cvtColor(mimg, cv2.COLOR_BGR2GRAY)

	# cv2.imwrite(im1[:-4]+"_gray.png", fimg_g)
	# cv2.imwrite(im2[:-4]+"_gray.png", mimg_g)

	fimgs.append(fimg_g)
	mimgs.append(mimg_g)
	(h, w) = fimg_g.shape
	while True:
		(x, y) = fimgs[-1].shape
		x = x // 2
		y = y // 2
		if x >= 3 and y >= 3:
			fimg = cv2.resize(fimgs[-1], (y, x))
			mimg = cv2.resize(mimgs[-1], (y, x))
			fimgs.append(fimg)
			mimgs.append(mimg)
			# cv2.imwrite(im1[:-4]+'_'+str(x)+'_'+str(y)+'.png', cv2.resize(fimg, (w, h), interpolation=cv2.INTER_NEAREST))
			# cv2.imwrite(im2[:-4]+'_'+str(x)+'_'+str(y)+'.png', cv2.resize(mimg, (w, h), interpolation=cv2.INTER_NEAREST))
			# cv2.imshow('', fimgs[-1])
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
		else:
			break
	trans_h = 0
	trans_w = 0
	trans = [[-1, -1], [-1, 0], [-1, 1] \
			, [0, -1], [0, 0], [0, 1], \
			[1, -1], [1, 0], [1, 1]]

	for t in range(len(fimgs)-1, -1, -1):
		(h, w) = fimgs[t].shape
		#print("begin", trans_h, trans_w)
		medf = np.median(fimgs[t])
		medm = np.median(mimgs[t])
		bitf = fimgs[t] > medf
		bitm = mimgs[t] > medm
		maskf = np.logical_or(fimgs[t] <= (medf-255*ignoreRatio), fimgs[t] >= (medf+255*ignoreRatio))
		maskm = np.logical_or(mimgs[t] <= (medm-255*ignoreRatio), mimgs[t] >= (medm+255*ignoreRatio))
		loss = np.zeros((3, 3))
		

		# tmp = np.zeros((h, w))
		# tmp[bitf] = 255
		# cv2.imwrite(im1[:-4]+'_bit_'+str(h)+'_'+str(w)+'.png', cv2.resize(tmp, (fimg_g.shape[1], fimg_g.shape[0]), interpolation=cv2.INTER_NEAREST))
		# tmp = np.zeros((h, w))
		# tmp[maskf] = 255
		# cv2.imwrite(im1[:-4]+'_mask_'+str(h)+'_'+str(w)+'.png', cv2.resize(tmp, (fimg_g.shape[1], fimg_g.shape[0]), interpolation=cv2.INTER_NEAREST))
		
		# tmp = np.zeros((h, w))
		# tmp[bitm] = 255
		# cv2.imwrite(im2[:-4]+'_bit_'+str(h)+'_'+str(w)+'.png', cv2.resize(tmp, (fimg_g.shape[1], fimg_g.shape[0]), interpolation=cv2.INTER_NEAREST))
		# tmp = np.zeros((h, w))
		# tmp[maskm] = 255
		# cv2.imwrite(im2[:-4]+'_mask_'+str(h)+'_'+str(w)+'.png', cv2.resize(tmp, (fimg_g.shape[1], fimg_g.shape[0]), interpolation=cv2.INTER_NEAREST))

		bitm = shift(bitm, trans_h, 0)
		bitm = shift(bitm, trans_w, 1)
		maskm = shift(maskm, trans_h, 0)
		maskm = shift(maskm, trans_w, 1)
		for i in range(3):
			for j in range(3):
				bit_sh = shift(bitm, i-1, 0)
				mask_sh = shift(maskm, i-1, 0)
				bit_sh = shift(bit_sh, j-1, 1)
				mask_sh = shift(mask_sh, j-1, 1)
				loss_matrix = np.logical_and(maskf, np.logical_and(mask_sh, np.logical_xor(bit_sh, bitf)))
				loss[i][j] = np.sum(loss_matrix)
				loss[i][j] += (abs(i-1)*w + abs(j-1)*h)*0.1
				tmp = np.zeros((h, w))
				tmp[loss_matrix] = 255
				# cv2.imwrite(im2[:-4]+'_lose_'+str(h)+'_'+str(w)+'_'+str(i)+'_'+str(j)+'.png', cv2.resize(tmp, (fimg_g.shape[1], fimg_g.shape[0]), interpolation=cv2.INTER_NEAREST))
				#print(i," ",j,loss[i][j])		
				#tmp=np.zeros(bitf.shape)
				#tmp[loss_matrix] = 255
				#tmp[bitf]=128
				#plt.imshow(tmp)
				#plt.show()
		m = np.argmin(loss)
		

		#print(m)
		trans_h += trans[m][0]
		trans_w += trans[m][1]

		if t > 0:
			trans_h *= 2
			trans_w *= 2

		#print("after", trans_h, trans_w)
	return [trans_h, trans_w]

if __name__ == '__main__':

	if len(sys.argv) != 5:
		print("usage: python3 mtb.py [imgs_dir] [imgs_list] [choose_mid] [out_dir]")
		exit(0)

	d = sys.argv[1]
	if d[-1] != '/':
		d = d+'/'

	d_out = sys.argv[4]
	if d_out[-1] != '/':
		d_out = d_out+'/'
	
	img_names = []
	out_names = []
	for i in open(sys.argv[2], 'r'):
		img_names.append(d+i[:-1])
		out_names.append(d_out+i[:-1])
	shifts = [[0, 0]]

	for i in range(len(img_names)-1):
		s = count_shift(img_names[i], img_names[i+1])
		shifts.append(s)
	
	# print(shifts)
	mid = int(sys.argv[3])

	n_shifts = np.zeros((len(shifts),  2), dtype=np.int)

	for i in range(mid-1, -1, -1):
		n_shifts[i][0] = n_shifts[i+1][0] - shifts[i+1][0]
		n_shifts[i][1] = n_shifts[i+1][1] - shifts[i+1][1]

	for i in range(mid+1, len(shifts)):
		n_shifts[i][0] = n_shifts[i-1][0] + shifts[i][0]
		n_shifts[i][1] = n_shifts[i-1][1] + shifts[i][1]

	# print(n_shifts)
	for i in range(len(img_names)):

		out = cv2.imread(img_names[i])
	
		trans_h = n_shifts[i][0]
		trans_w = n_shifts[i][1]
		# print(trans_h, trans_w)
		if trans_h != 0:
			out = np.roll(out, trans_h, axis=0)
			if trans_h > 0:
				out[:trans_h, :] = [255, 0, 0]
			else:
				out[trans_h:, :] = [255, 0, 0]

		if trans_w != 0:
			out = np.roll(out, trans_w, axis=1)
			if trans_w > 0:
				out[:, :trans_w] = [255, 0, 0]
			else:
				out[:, trans_w:] = [255, 0, 0]

		cv2.imwrite(out_names[i], out)
