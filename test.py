"""
"""
import argparse
import numpy as np
import cv2
from face_recognizer import FaceRecognizer
import time

###
parser = argparse.ArgumentParser(description='face_recognization')
parser.add_argument('--face_db_root', type=str, default='data/mask_nomask', help='the root path of target database')
parser.add_argument('--input_video_path', type=str, default='D:/mask_data/02.mp4', help='the path of input video')
parser.add_argument('--output_video_path', type=str, default='output.mp4', help='the path of input video')
###
args = parser.parse_args()
recognizer = FaceRecognizer()
recognizer.create_known_faces(args.face_db_root)
frame_count = 0
#recognizer.test_100x()


#input_movie = cv2.VideoCapture(args.input_video_path)
input_movie = cv2.VideoCapture(0)
#video_size = (1024, 720)
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#output_movie = cv2.VideoWriter(args.output_video_path, fourcc, 10, video_size)

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_count += 1
    # Quit when the input video file ends
    if not ret: break
    if(frame_count % 15 == 0):
        #frame = cv2.resize(frame, dsize=video_size)
        # frame = frame[100:800, 900: 1800]
        frame = cv2.flip(frame, 1)
        item = recognizer.recognize(frame, 0.5)
        if item:
            name, (left, top, right, bottom), _, score = item
            print(name, score)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, "%s %.3f" % (name, score), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        #output_movie.write(frame)
        cv2.imshow('0', frame)
        cv2.waitKey(1)

# All done!
#output_movie.release()
input_movie.release()
cv2.destroyAllWindows()


