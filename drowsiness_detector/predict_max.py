import torch
import cv2
import os
import logging
import mediapipe as mp
import torch
import torch.nn.functional as F
import copy 
# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_facemesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

def add_landmarks(img, landmarks):
    h, w = img.shape[:2]
    output = img.copy()
    spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(255, 255, 255))
    mp_drawing.draw_landmarks(image=output, landmark_list=landmarks,
                              connections=mp_facemesh.FACEMESH_TESSELATION,
                              connection_drawing_spec=spec)
    for idx in all_idxs:
        x = int(landmarks.landmark[idx].x * w)
        y = int(landmarks.landmark[idx].y * h)
        cv2.circle(output, (x, y), 3, (255, 255, 255), -1)
    return output

def preprocess_single_image(image, save_img=False, img_size=145):
    if image is None:
        logging.warning(f"Image read failed: {image}")
        return None

    h, w = image.shape[:2]
    results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.detections:
        logging.warning(f"No face detected")
        return None

    bbox = results.detections[0].location_data.relative_bounding_box
    x1 = max(0, int(bbox.xmin * w))
    y1 = max(0, int(bbox.ymin * h))
    x2 = min(w, x1 + int(bbox.width * w))
    y2 = min(h, y1 + int(bbox.height * h))

    margin = 0.1
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(w, x2 + dx), min(h, y2 + dy)
    roi = image[y1:y2, x1:x2]

    if min(roi.shape[:2]) < 50:
        logging.warning(f"ROI too small: {roi.shape}")
        return None

    with mp_facemesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as mesh:
        mesh_res = mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    if not mesh_res.multi_face_landmarks:
        logging.warning(f"No landmarks found")
        return None

    annotated = add_landmarks(roi.copy(), mesh_res.multi_face_landmarks[0])
    return annotated
class FatiguePredictor:
    def __init__(self):
        self.model = torch.load('drowsiness_detector/model/model_real_time (1).pt' , weights_only= False)
    def predict(self,image, device=torch.device('cuda')) -> torch.Tensor:
        model =self.model
        print(type(image))
        print("Hello")
        processed = preprocess_single_image(image)
        processed_show = copy.deepcopy(processed)
        try:
            if processed == None:
                print("No face was detected")
                return torch.tensor([0,0]) , image
        except Exception as e:
            print("Face Detected")
        if not isinstance(processed, torch.Tensor):
            processed = torch.tensor(processed, dtype=torch.float32)
        processed = processed.permute(2, 0, 1)
        processed = processed.unsqueeze(0)
        
        model = model.to(device)
        x = processed.to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs  = F.softmax(logits, dim=1)
        return probs.squeeze(0).cpu() , processed_show
if __name__ == '__main__':
    img = cv2.imread('Drowsiness-Landmark-Detection\drowsiness_detector\image.png')
    predictor = FatiguePredictor()
    out = predictor.predict(img)
    print(out)
