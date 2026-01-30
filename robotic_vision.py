import cv2
import numpy as np
import random

img1 = cv2.imread("frame1.jpg")
img2 = cv2.imread("frame2.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

blur1 = cv2.GaussianBlur(gray1,(25,25),0)
blur2 = cv2.GaussianBlur(gray2,(25,25),0)

h1, w1 = blur1.shape
h2, w2 = blur2.shape

# pick a random center that can fit a 3x3 region
# y1 = random.randint(1, h1 - 2)
# x1 = random.randint(1, w1 - 2)

# y2 = random.randint(1, h2 - 2)
# x2 = random.randint(1, w2 - 2)

y1, x1 = h1 // 2, w1 // 2
y2, x2 = h2 // 2, w2 // 2

region1 = gray1[y1-2:y1+3, x1-2:x1+3]
region2 = gray2[y2-2:y2+3, x2-2:x2+3]

It = region2.astype(np.float32) - region1.astype(np.float32)

def optical_flow_error(Ix, Iy, It, u, v):
    error = Ix*u + Iy*v + It
    return np.mean(error**2)

print(f"center pixel: ({y1}, {x1})")
print(region1)

print(f"center pixel: ({y2}, {x2})")
print(region2)

print(f"temporal gradient: \n")
print(It)

motions = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]

Ix = np.zeros((5,5), dtype=np.float32)
Iy = np.zeros((5,5), dtype=np.float32)

Ix[:,2] = (region1[:,3] - region1[:,1]) / 2.0
Iy[2,:] = (region1[3,:] - region1[1,:]) / 2.0

for u, v in motions:
    e = optical_flow_error(Ix, Iy, It, u, v)
    print(f"u={u}, v={v}, MSE={e:.4f}")