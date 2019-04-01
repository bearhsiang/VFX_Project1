import cv2
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import math

t = []
imgNames = []
imgs = []
gray_imgs = []
n = 1000
points = []

def weight(a):
    if(a > 128):
        if(a == 255): return 1
        return 255-a
    return a

def m(a):
    if(a == [255, 0, 0]):
        return True
    return False

def g_trans(a, time):
    return g[a]-np.log(t[time])

def findCurve_gray(images):
    p = len(images)
    A = np.zeros((n*p+255, 256+n))
    b = np.zeros(n*p+255)

    k = 0
    for i in range(n):
        for j in range(p):
            m = images[j].item(points[i][0], points[i][1])
            A[k, m] = weight(m)
            A[k, 256+i] = -weight(m)
            b[k] = weight(m)*math.log(t[j])
            k += 1

    A[k, 127] = 1
    k += 1

    for i in range(254):
        A[k, i] = weight(i+1)
        A[k, i+1] = -2*weight(i+1)
        A[k, i+2] = weight(i+1)
        k += 1

    x = np.linalg.lstsq(A, b)[0]

    return x[0:256], x[256:]

def findCurve_color(images):
    print("find response curve of channel")
    p = len(images)
    A = np.zeros((n*p*3+255, 256+n))
    b = np.zeros(n*p*3+255)

    k = 0
    for i in range(n):
        for j in range(p):
            for c in range(3):
                m = images[j].item(points[i][0], points[i][1], c)
                A[k, m] = weight(m)
                A[k, 256+i] = -weight(m)
                b[k] = weight(m)*math.log(t[j])
                k += 1

    A[k, 127] = 1
    k += 1

    for i in range(254):
        A[k, i] = weight(i+1)
        A[k, i+1] = -2*weight(i+1)
        A[k, i+2] = weight(i+1)
        k += 1

    x = np.linalg.lstsq(A, b)[0]

    return x[0:256], x[256:]


if len(sys.argv) != 4:
    print("usage: python3 imgsTohdr.py [img_dir] [img_list] [hdrName]")
    exit(0)

d = sys.argv[1]

if d[-1] != '/':
    d = d+'/'

for i in open(sys.argv[2], "r"):
    [imgname, s] = i.split(' ')
    t.append(eval(s))
    imgNames.append(imgname)
    img = cv2.imread(d+imgname)
    # img = cv2.resize(img, (800, 600))
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs.append(img)
    # gray_imgs.append(gray_image)

(h, w, c) = imgs[0].shape

np.random.seed(int(time.time()))

# cv2.imshow("", imgs[8])
# cv2.waitKey()

record = np.zeros(256)
i = 0
while i < n or (np.sum(record) < 230 and i < 2*n):
    p1 = np.random.randint(h)
    p2 = np.random.randint(w)
    hit = 0
    for j in range(len(imgs)):
        if imgs[j].item(p1, p2, 0) == 255 and imgs[j].item(p1, p2, 1) == 0 \
            and imgs[j].item(p1, p2, 2) == 0:
            hit = 1
            break
    if hit == 0:
        i += 1
        points.append((p1, p2))
        for k in range(len(imgs)):
            for c in range(3):
                record[imgs[k][p1][p2][c]] = 1

print(np.sum(record))
# for i in range(len(points)):
#     plt.plot(points[i][0], points[i][1], 'ro')

# plt.show()
# while True:
#
#     if np.sum(record) != 256:
#         print(np.sum(record))
#         points.append((np.random.randint(0.05*h, 0.95*h), np.random.randint(0.05*w, 0.95*w)))
#         record[imgs[3].item(points[-1][0], points[-1][1])] = 1
#     else:
#         break

g = np.zeros(256)

[g, tmp] = findCurve_color(imgs)


# [g[3], E[3]] = findCurve_gravfxy(gray_imgs)


# print(g, len(g))
# print(E, len(E))
#
# color = ['bo', 'go', 'ro', 'co']
# for i in range(len(g)):
#    plt.plot(i, g[i], color[0])

# plt.show()

eImg = np.zeros((h, w, 3))
eImg_w = np.zeros((h, w, 3))
eImg_log = np.zeros((h, w, 3))

w_vec = np.vectorize(weight)
mask_vec = np.vectorize(m)
g_vec = np.vectorize(g_trans)

for i in range(len(imgs)):
    mask = np.all(imgs[i] == [255, 0, 0], axis=-1)
    
    for j in range(3):
        w_matric = w_vec(imgs[i][:, :, j])
        w_matric[mask] = 0
        eImg_w[:, :, j] += w_matric
        eImg_log[:, :, j] += w_matric * g_vec(imgs[i][:, :, j], i)

check = eImg_w == 0
eImg_w[check] = 1
eImg_log = eImg_log / eImg_w
eImg = np.exp(eImg_log)
eImg[check][:] = 0


# f, (ax1, ax2) = plt.subplots(1, 2)

# im1 = ax1.pcolormesh(eImg[:, :, 0], cmap="gist_rainbow")
# f.colorbar(im1, ax=ax1)

# im2 = ax2.pcolormesh(eImg[:, :, 1], cmap="gist_rainbow")
# f.colorbar(im2, ax = ax2)

# plt.show()



# g = cv2.cvtColor(eImg_log, cv2.COLOR_BGR2GRAY)
eImg = np.float32(eImg)
# gray_eImg = np.float32(gray_eImg)

# cv2.imwrite(sys.argv[3]+".exr", eImg)
# cv2.imwrite(sys.argv[3]+".exr", gray_eImg)


cv2.imwrite(sys.argv[3]+".hdr", eImg)
# cv2.imwrite(sys.argv[3]+".hdr", gray_eImg)
# # low_image = extract(gray_eImg, 0.01)
# # cv2.imwrite("testLow.jpg", low_image)

# np.save(sys.argv[2], eImg)
# np.save(sys.argv[3], gray_eImg)

# print(np.max(eImg), np.min(eImg))



# tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
# ldrDrago = tonemapDrago.process(eImg)
# ldrDrago = 3 * ldrDrago
# cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)
# fig, (ax0, ax1) = plt.subplots(1, 2)
# im = ax0.pcolormesh(eImg)
# fig.colorbar(im, ax=ax0)
# im = ax1.pcolormesh(eImg_log)
# fig.colorbar(im, ax=ax1)
#
# plt.show()
