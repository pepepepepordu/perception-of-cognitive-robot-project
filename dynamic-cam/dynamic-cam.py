from controller import Robot
import cv2
import numpy as np

# --- 1. CONSTANTS & SETUP ---
TIMESTEP = 64
WINDOW_SIZE = 5       # "Local 5x5 image patch"
HALF_WIN = WINDOW_SIZE // 2
GRADIENT_THRESHOLD = 5.0 # Threshold to discard unstable points

# "Discrete set of candidate motion vectors"
CANDIDATE_VECTORS = [
    (0, 0),   # Stationary
    (-1, 0),  # Left
    (1, 0),   # Right
    (0, -1),  # Up
    (0, 1)    # Down
]

# Initialize Robot and Camera
robot = Robot()
camera = robot.getDevice('camera')
camera.enable(TIMESTEP)

# Initialize Motors (Stationary)
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

print("Starting PDF-aligned Vision Controller...")

prev_gray = None

# --- 2. MAIN LOOP ---
while robot.step(TIMESTEP) != -1:
    # A. Capture Image
    raw_image = camera.getImage()
    if not raw_image: continue

    width = camera.getWidth()
    height = camera.getHeight()
    img_np = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
    
    # B. Pre-processing
    curr_gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
    curr_gray = cv2.GaussianBlur(curr_gray, (3, 3), 0)

    if prev_gray is None:
        prev_gray = curr_gray
        continue

    # C. Feature Selection (Shi-Tomasi)
    features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=20,
        qualityLevel=0.01,
        minDistance=5
    )

    # Visualization Copy
    vis_image = img_np.copy()
    
    if features is not None:
        features = np.int8(features)
        
        for f in features:
            x, y = f.ravel()

            # Ensure 5x5 patch is inside image boundaries
            if x - HALF_WIN < 1 or x + HALF_WIN >= width - 1 or \
               y - HALF_WIN < 1 or y + HALF_WIN >= height - 1:
                continue

            # D. Gradient Computation
            patch_prev = prev_gray[y-HALF_WIN:y+HALF_WIN+1, x-HALF_WIN:x+HALF_WIN+1].astype(float)
            patch_curr = curr_gray[y-HALF_WIN:y+HALF_WIN+1, x-HALF_WIN:x+HALF_WIN+1].astype(float)

            # Spatial Gradients (Central Difference)
            prev_x_plus = prev_gray[y-HALF_WIN:y+HALF_WIN+1, x-HALF_WIN+1:x+HALF_WIN+2].astype(float)
            prev_x_minus = prev_gray[y-HALF_WIN:y+HALF_WIN+1, x-HALF_WIN-1:x+HALF_WIN].astype(float)
            Ix = (prev_x_plus - prev_x_minus) / 2.0

            prev_y_plus = prev_gray[y-HALF_WIN+1:y+HALF_WIN+2, x-HALF_WIN:x+HALF_WIN+1].astype(float)
            prev_y_minus = prev_gray[y-HALF_WIN-1:y+HALF_WIN, x-HALF_WIN:x+HALF_WIN+1].astype(float)
            Iy = (prev_y_plus - prev_y_minus) / 2.0

            # Temporal Gradient
            It = patch_curr - patch_prev

            # Discard weak gradients
            grad_magnitude = np.mean(np.abs(Ix) + np.abs(Iy))
            if grad_magnitude < GRADIENT_THRESHOLD:
                continue

            # E. Evaluate Candidate Vectors (MSE)
            best_mse = float('inf')
            best_u, best_v = 0, 0

            for u, v in CANDIDATE_VECTORS:
                error_term = (Ix * u + Iy * v + It)
                mse = np.mean(error_term ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_u, best_v = u, v

            # F. Classification & Visualization
            color = (0, 255, 0) # Green (Static)
            
            if best_u != 0 or best_v != 0:
                color = (0, 0, 255) # Red (Dynamic)
                print(f"Feature ({x},{y}): Motion ({best_u}, {best_v}) -> DYNAMIC")
            else:
                # --- CHANGE 1: ADDED STATIC PRINT ---
                print(f"Feature ({x},{y}): Motion (0, 0) -> STATIC")
            
            # Draw dot
            cv2.circle(vis_image, (x, y), 3, color, -1)
            # Draw arrow if moving
            if best_u != 0 or best_v != 0:
                 cv2.line(vis_image, (x, y), (x + best_u*10, y + best_v*10), color, 2)

    # Update previous frame
    prev_gray = curr_gray

    # --- CHANGE 2: MAKE WINDOW BIGGER ---
    # Resize image to 5x original size using Nearest Neighbor (keeps pixels sharp)
    scale_factor = 10
    big_vis = cv2.resize(vis_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("PDF Method: Optical Flow", big_vis)
    cv2.waitKey(1)