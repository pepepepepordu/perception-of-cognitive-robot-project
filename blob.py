import cv2
import numpy as np

img1 = cv2.imread("frame1.jpg")  
img2 = cv2.imread("frame2.jpg")  

cvtimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cvtimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

subtracted = cv2.subtract(cvtimg1, cvtimg2)

gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

thresh = cv2.bitwise_not(thresh)

kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

num_labels, labels = cv2.connectedComponents(closed)

print("Number of blobs:", num_labels - 1)

blob_img = np.zeros_like(subtracted)

for label in range(1, num_labels):
    mask = labels == label
    color = np.random.randint(0, 255, size=3)
    blob_img[mask] = color

cv2.imshow("Original", subtracted)
cv2.imshow("Threshold Edge", thresh)
cv2.imshow("Closed Edge", closed)
cv2.imshow("Blob Separation", blob_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
