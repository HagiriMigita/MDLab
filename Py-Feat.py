import pandas as pd
import numpy as np
import cv2
import re
import os
import matplotlib.pyplot as plt
from PIL import Image

from feat import Detector
face_model = "retinaface"
landmark_model ="mobilenet"
au_model = "svm"
emotion_model = "resmasknet"
detector = Detector(
    face_model = face_model,
    landmark_model = landmark_model,
    au_model = au_model,
    emotion_model = emotion_model
)

from feat.utils import get_test_data_path

