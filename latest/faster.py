import os
import argparse
import time

from PIL import Image
import numpy as np
import cv2

import torch
from torchvision import transforms

from networks.dan import DAN
import mediapipe as mp

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        checkpoint = torch.load('./checkpoints/affecnet8_epoch5_acc0.6209.pth',
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        results = self.face_detection.process(img)
        if not results.detections:
            return []
        
        faces = []
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append((x, y, w, h))
        
        return faces

    def get_eyes_center(self, img):
        img_rgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        results = self.face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return (0, 0)

        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = (landmarks[33].x, landmarks[33].y)  # 左目中心
        right_eye = (landmarks[263].x, landmarks[263].y)  # 右目中心

        ih, iw, _ = img_rgb.shape
        eyes_center_x = int((left_eye[0] + right_eye[0]) * iw / 2)
        eyes_center_y = int((left_eye[1] + right_eye[1]) * ih / 2)

        return (eyes_center_x, eyes_center_y)

    def fer(self, image):
        img0 = cv2pil(image)
        faces = self.detect(img0)

        if len(faces) == 0:
            return 'null'

        x, y, w, h = faces[0]
        img = img0.crop((x, y, x+w, y+h))
        img = self.data_transforms(img)
        img = img.view(1, 3, 224, 224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out, 1)
            index = int(pred)
            label = self.labels[index]

            return label

if __name__ == "__main__":

    model = Model()

    cap = cv2.VideoCapture(0)
    fireworks_image = cv2.imread('happiness.png', cv2.IMREAD_UNCHANGED)
    sadness_image = cv2.imread('sadness.png', cv2.IMREAD_UNCHANGED)
    anger_effect_path = "anger.mp4"
    anger_cap = cv2.VideoCapture(anger_effect_path)

    if fireworks_image is None:
        print("Error: happiness.png could not be loaded. Please check the file path.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    if sadness_image is None:
        print("Error: sadness.png could not be loaded. Please check the file path.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    if not anger_cap.isOpened():
        print("Error: anger.mp4 could not be loaded. Please check the file path.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    sadness_image = cv2.cvtColor(sadness_image, cv2.COLOR_BGRA2RGBA)

    def paste_image(base_image, overlay_image, x_offset, y_offset):
        h, w = overlay_image.shape[:2]
        base_h, base_w = base_image.shape[:2]

        if x_offset < 0:
            overlay_image = overlay_image[:, -x_offset:]
            w = overlay_image.shape[1]
            x_offset = 0
        if y_offset < 0:
            overlay_image = overlay_image[-y_offset:, :]
            h = overlay_image.shape[0]
            y_offset = 0
        if x_offset + w > base_w:
            overlay_image = overlay_image[:, :base_w - x_offset]
            w = overlay_image.shape[1]
        if y_offset + h > base_h:
            overlay_image = overlay_image[:base_h - y_offset, :]
            h = overlay_image.shape[0]

        alpha_s = overlay_image[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            base_image[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
            (alpha_s * overlay_image[:, :, c] + alpha_l * base_image[y_offset:y_offset+h, x_offset:x_offset+w, c])

    def paste_fireworks(image, fireworks, num_fireworks=3):
        for _ in range(num_fireworks):
            y_offset = np.random.randint(0, image.shape[0] - fireworks.shape[0])
            x_offset = np.random.randint(0, image.shape[1] - fireworks.shape[1])
            paste_image(image, fireworks, x_offset, y_offset)

    with torch.no_grad():
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            _, frame = cap.read()
            label = model.fer(frame)

            print(label)

            effect_duration = 60

            frame_flags = frame.flags.writeable
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if 'happy' in label:
                frame_flags = True
                paste_fireworks(frame, fireworks_image)

            if 'anger' in label:
                ret, anger_frame = anger_cap.read()
                if not ret:
                    anger_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, anger_frame = anger_cap.read()

                anger_frame = cv2.cvtColor(anger_frame, cv2.COLOR_BGR2BGRA)
                hsv = cv2.cvtColor(anger_frame, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35, 100, 100])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                anger_frame[mask != 0, 3] = 0

                anger_frame = cv2.cvtColor(anger_frame, cv2.COLOR_BGRA2RGBA)
                frame_flags = True
                paste_image(frame, anger_frame, 0, 0)

            if 'sad' in label:
                eyes_center = model.get_eyes_center(cv2pil(frame))
                frame_flags = True
                paste_image(frame, sadness_image, eyes_center[0] - sadness_image.shape[1] // 2, eyes_center[1] - sadness_image.shape[0] // 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0))

            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                print(f"Frames per second: {frame_count}")
                frame_count = 0
                start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()