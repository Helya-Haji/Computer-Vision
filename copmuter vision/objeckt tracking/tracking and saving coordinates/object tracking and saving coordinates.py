import cv2
import numpy as np

template = cv2.imread(r'E:\work\task-1\book.jpg', cv2.IMREAD_COLOR)  # Ensure color loading
if template is None:
    print("Error: Template image not found. Check the path.")
    exit(1)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)   # Convert the template to grayscale for feature detection
keypoints_template, descriptors_template = sift.detectAndCompute(template_gray, None)   # Detecting keypoints in image

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit(1)

# Create or open the text file for saving coordinates (use raw string for file path)
file_path = r"C:\Users\Helya\Desktop\git\copmuter vision\objeckt tracking\object_positions.txt"
with open(file_path, 'w') as file:
    file.write('Frame, X, Y\n')  # Write headers
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame from video.")
            break
        
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Converting the frame to grayscale for feature detection
        
        keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)   # Detecting keypoints in frame
        
        matches = bf.knnMatch(descriptors_template, descriptors_frame, k=2)   # Matching image and frame
        
        good_matches = []   # Filtering matches
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        if len(good_matches) >= 4:
            src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)   # Computing homography matrix
            
            if M is not None:   # Drawing bounding box
                h, w, _ = template.shape
                template_corner = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                transformed_corner = cv2.perspectiveTransform(template_corner, M)
                frame = cv2.polylines(frame, [np.int32(transformed_corner)], True, (0, 255, 0), 2)
                
                # Calculate the center of the bounding box
                center_x = np.mean(transformed_corner[:, 0, 0])
                center_y = np.mean(transformed_corner[:, 0, 1])

                # Write the frame number and center coordinates to the file
                file.write(f'{frame_count}, {center_x}, {center_y}\n')
                
                cv2.imshow('Bounding Box', frame)   # Displaying the frame with bounding box
            else:
                print("Homography could not be computed.")
        
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()        
cv2.destroyAllWindows()

print(f"Coordinates saved to {file_path}")
