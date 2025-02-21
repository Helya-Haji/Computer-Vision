import cv2
import numpy as np

cap = cv2.VideoCapture(0)

hsv_file = r"C:\Users\Helya\Desktop\git\copmuter vision\picking HSV color from video\hsv_colors.txt"   #  creating Text file to save HSV values and the range

hsv_values = []


def pick_color(event, x, y, flags, param):  # Function to handle mouse click events
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        bgr_color = frame[y, x]   # Get the color at the clicked pixel in BGR format
        
        # Convert BGR to HSV
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)
        hsv_color = hsv_color[0][0]

        hsv_values.append(hsv_color)

        # Calculate min and max HSV ranges
        hsv_array = np.array(hsv_values)
        min_hsv = hsv_array.min(axis=0)
        max_hsv = hsv_array.max(axis=0)

        # Save the HSV color and range to a text file
        with open(hsv_file, "a") as file:
            file.write(f"Clicked HSV: {hsv_color.tolist()}\n")
            file.write(f"Current HSV Range: Lower - {min_hsv.tolist()}, Upper - {max_hsv.tolist()}\n")
            file.write("----\n")  # Divider for readability
        
        # Print the HSV value and range to the console
        print(f"HSV Color at ({x}, {y}): {hsv_color}")
        print(f"Saved HSV: {hsv_color.tolist()}")
        print(f"Current HSV Range: Lower - {min_hsv.tolist()}, Upper - {max_hsv.tolist()}")

# Set up the OpenCV window and mouse callback function
cv2.namedWindow("Marker Detection")
cv2.setMouseCallback("Marker Detection", pick_color)

print("Click on the marker in the video to capture its HSV color. Press 'q' to quit.")

while True:
    # Capture each frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display the frame
    cv2.imshow("Marker Detection", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
