import sys
import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv.VideoCapture(0)
scaling_factor = 1

while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit("Couldn't Load frame")
    frame = cv.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_CUBIC)
    rows, cols = frame.shape[:2]
    src_points = np.float32([[0, 0], [cols-1, 0], [0,rows-1]])
    dst_points = np.float32([[cols-1, 0], [0, 0], [cols-1, rows-1]])
    
    AffineMatrix = cv.getAffineTransform(src_points, dst_points)
    frame = cv.warpAffine(frame, AffineMatrix, (cols, rows))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=1)
    '''
    scaleFactor : if we don;t find and image in the current scale, the next size to check will be,
    in our case 1.3 times bigger than the current size,
    minNeighbors : it is the thresold the specifies the minimum number of adjacent rectangles needed
    to keep the current rectangle, it can be used to increased the robustness of the face detector.
    
    in case face recognition doesn't work as expected, reducing the threshold value to obtain better recognition.
    In cases where the images suffers some delay due to processing the detection, reduce the size of the
    scaled frame by 0.4 or 0.3 
    
    '''
    for (x, y, w, h) in face_rects:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv.putText(frame, "X", (x, 0), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1 )
    cv.imshow("Face Detector", frame)
    # cv.imshow("Face",face)
    c = cv.waitKey(1)
    if c == 27:
        break
cap.release()
cv.destroyAllWindows()
