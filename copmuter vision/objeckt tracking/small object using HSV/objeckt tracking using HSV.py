import cv2
import numpy as np

lower_hsv = np.array([26, 79, 106])  # Lower bound of the HSV range
upper_hsv = np.array([33, 220, 215])  # Upper bound of the HSV range

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit(1)

file_path = r"C:\Users\Helya\Desktop\git\copmuter vision\objeckt tracking\small object using HSV\object_positions.txt"
with open(file_path, 'w') as file:
    file.write('Frame, X, Y\n')  # Write headers
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame from video.")
            break
        
        frame_count += 1

        # Apply Gaussian blur and convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0)

        # Create a mask based on the specified HSV color range
        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

        # Morphological transformations (erosion + dilation) to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get the contour area
            area = cv2.contourArea(contour)
            
            # Only proceed if the area is smaller than the threshold (to focus on small objects)
            if area > 50 and area < 600:  # Adjust this range as needed
                # Get the bounding box and draw it
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Optionally, fit a circle around the contour
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                center = (int(cx), int(cy))
                radius = int(radius)

                # Save the center coordinates of the circle
                center_x, center_y = int(cx), int(cy)
                
                # Write the frame number and center coordinates to the file
                file.write(f'{frame_count}, {center_x}, {center_y}\n')

                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame,  
                f'{frame_count}, {center_x}, {center_y}\n',  
                (50, 50),  
                font, 0.7,  
                (0, 0, 0),  
                2,  
                cv2.LINE_4)


        # Show the frame with the detected object and bounding box
        cv2.imshow("Object Tracking", frame)

        # Exit condition (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all OpenCV windows
cap.release()        
cv2.destroyAllWindows()

print(f"Coordinates saved to {file_path}")

