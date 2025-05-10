
import cv2 
import random
circle_radius = 20


vid = cv2.VideoCapture(0)
frame_count = 0  
update_interval = 10

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(True):
    ret, frame = vid.read()
    
    if not ret:
        print('the frame not found!')
        break

    if frame_count % update_interval == 0: 
        circle_center = (
            random.randint(circle_radius, width - circle_radius),
            random.randint(circle_radius, height - circle_radius)
        )


    cv2.circle(frame, circle_center, circle_radius, (0, 0, 255), -1)
    
    cv2.imshow('Real-Time Video with Circle', frame)

    frame_count += 1

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()        
cv2.destroyAllWindows()










