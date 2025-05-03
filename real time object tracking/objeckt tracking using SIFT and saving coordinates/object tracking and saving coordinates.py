import cv2
import numpy as np

template = cv2.imread(r"C:\\Users\\Helya\\Desktop\\computer-vision\\objeckt tracking\\objeckt tracking using SIFT and saving coordinates\\WIN_20250503_14_46_18_Pro.jpg", cv2.IMREAD_COLOR)  # Ensure color loading
if template is None:
    print("Error: Template image not found. Check the path.")
    exit(1)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  
keypoints_template, descriptors_template = sift.detectAndCompute(template_gray, None) 

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit(1)


file_path = r"C:\\Users\\Helya\\Desktop\\computer-vision\\objeckt tracking\\objeckt tracking using SIFT and saving coordinates\\object_positions.txt"
with open(file_path, 'w') as file:
    file.write('Frame, X, Y\n')  
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame from video.")
            break
        
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
        
        keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None) 
        
        matches = bf.knnMatch(descriptors_template, descriptors_frame, k=2)  
        
        good_matches = []   
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        if len(good_matches) >= 10:
            src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)  
            
            if M is not None: 
                h, w, _ = template.shape
                template_corner = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                transformed_corner = cv2.perspectiveTransform(template_corner, M)
                frame = cv2.polylines(frame, [np.int32(transformed_corner)], True, (0, 255, 0), 2)
             
                center_x = np.mean(transformed_corner[:, 0, 0])
                center_y = np.mean(transformed_corner[:, 0, 1])

                file.write(f'{frame_count}, {center_x}, {center_y}\n')
                
                cv2.imshow('Bounding Box', frame)  
            else:
                print("Homography could not be computed.")
        
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()        
cv2.destroyAllWindows()

print(f"Coordinates saved to {file_path}")
