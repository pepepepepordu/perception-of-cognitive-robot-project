import cv2
import numpy as np


img1 = cv2.imread("frame1.jpg")  
img2 = cv2.imread("frame2.jpg")  

cvtimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cvtimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

subtracted = cv2.subtract(cvtimg1, cvtimg2)

blur = cv2.GaussianBlur(subtracted, (25, 25), 10)

_, thresh1 = cv2.threshold(
    cvtimg1, 120, 255, cv2.THRESH_BINARY
)

_, thresh2 = cv2.threshold(
    cvtimg2, 120, 255, cv2.THRESH_BINARY
)

thresh1 = cv2.bitwise_not(thresh1)
thresh2 = cv2.bitwise_not(thresh2)

cv2.imshow("Threshold Edge Frame 1", thresh1)
cv2.imshow("Threshold Edge Frame 2", thresh2)

#kernel = np.ones((5, 5), np.uint8)
#closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

#cv2.imwrite("subtracted_image.png", subtracted)

#cv2.imshow("Subtracted Image", subtracted)

cv2.waitKey(0)
cv2.destroyAllWindows()
