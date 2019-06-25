import cv2
import numpy as np
import os

detect_face = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
BLUE = (255, 0, 0)  # BGR
root = cv2.VideoCapture(0)
while True:

    # capture frames
    ret, frame = root.read()

    # convert to grey scale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face = detect_face.detectMultiScale(grey, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in face:
        print(x, y, h)

        # test if it detect the face
        # roi_gray = grey[y:y + h, x:x + w]
        # img_item = "image.png"
        # cv2.imwrite(img_item, roi_gray)

        # crating ROI Bound
        rec_color = BLUE  # ROI color
        rec_thickness = 2  # how thick should the frame be
        rec_end_x = x+w   # end coordinate for x
        rec_end_y = y+h  # end coordinate for y

        cv2.rectangle(frame, (x, y), (rec_end_x, rec_end_y), rec_color, rec_thickness)


    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# close capture when done
root.release()
cv2.destroyAllWindows()
