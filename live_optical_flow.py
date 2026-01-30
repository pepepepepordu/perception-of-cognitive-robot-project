import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (25,25), 0)

h, w = prev_gray.shape

def optical_flow_error(Ix, Iy, It, u, v):
    error = Ix*u + Iy*v + It
    return np.mean(error**2)

motions = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25,25), 0)

    features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=3,
        qualityLevel=0.01,
        minDistance=20
    )

    if features is None:
        continue

    points = [(int(pt[0][1]), int(pt[0][0])) for pt in features]

    for idx, (y, x) in enumerate(points):
        region1 = prev_gray[y-2:y+3, x-2:x+3]
        region2 = gray[y-2:y+3, x-2:x+3]

        It = region2.astype(np.float32) - region1.astype(np.float32)

        Ix = np.zeros((5,5), dtype=np.float32)
        Iy = np.zeros((5,5), dtype=np.float32)

        Ix[:,2] = (region1[:,3] - region1[:,1]) / 2.0
        Iy[2,:] = (region1[3,:] - region1[1,:]) / 2.0

        best_error = float("inf")
        best_motion = (0, 0)

        if np.mean(Ix**2 + Iy**2) < 1e-3:
            print(f"Point {idx+1}: weak gradient → skipped")
            continue

        for u, v in motions:
            e = optical_flow_error(Ix, Iy, It, u, v)

            if e < best_error:
                best_error = e
                best_motion = (u, v)

            print(f"Point {idx+1} | u={u}, v={v}, MSE={e:.4f}")
    
        u_best, v_best = best_motion

        if u_best == 0 and v_best == 0:
            direction = "No movement"
        elif u_best == 1:
            direction = "Right"
        elif u_best == -1:
            direction = "Left"
        elif v_best == 1:
            direction = "Down"
        elif v_best == -1:
            direction = "Up"

        print(f"➡ Point {idx+1} motion: {direction} (u={u_best}, v={v_best}, MSE={best_error:.4f})\n")

        cv2.rectangle(frame, (x-2, y-2), (x+2, y+2), (0,0,255), 1)

    cv2.imshow("Live Optical Flow Debug", frame)

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('a'):  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()