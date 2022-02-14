"""
"""
import argparse
import time
from datetime import datetime
import cv2


video = cv2.VideoCapture(0)
while True:
    _, frame = video.read()
    cv2.imshow('0',frame)
    cv2.waitKey(1)

