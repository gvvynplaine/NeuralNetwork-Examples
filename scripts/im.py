import numpy as np
import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print flags

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        name = "capture" + stamp + ".jpeg"
        cv2.imwrite(name,gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
