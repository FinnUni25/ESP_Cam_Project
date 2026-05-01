import cv2
import numpy as np
from pathlib import Path
import json
import math

model_name = "face_detection_yunet_2023mar.onnx"
images_path = Path(__file__).parent / "images"
model_path = Path(__file__).parent / "models" / model_name

def rotate_image(image, center, angle):
    matrix = cv2.getRotationMatrix2D(center = center, angle = angle, scale = 1)
    h, w = image.shape[:2]
    image_alligned = cv2.warpAffine(src = image, M = matrix, dsize = (w, h), flags=cv2.INTER_CUBIC)  # inter_cubic liefert schärfere Kanten was für Gesichtserkennung wichtig ist
    return image_alligned


detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (100, 100),   # irgendwelche Werte für die Input size, wird beim bildlesen sowieso automatisch erkannt
        score_threshold = 0.7,
        nms_threshold=0.3,   # filter out overlapping bboxes (iou threshold)
        top_k = 100,  # how many boxes are selected before nms filtering
    )

bboxes = {
    "images": [],
    "annotations": [],
    "persons": [],
}

#### Struktur der finalen json:
# bboxes = {
#     images: [{
#         image_name: ,
#         image_id: ,
#         width: ,
#         height: ,
#     }],
#     annotations: [{
#         image_id: ,
#         annotation_id: ,
#         bbox: [x, y, w, h],
#         person_id: ,
#     }],
#     persons: [{
#         person_id: ,
#         person_name: ,
#     }]
# }
img_id = 0
ann_id = 0
pers_id = 0

bboxes["persons"].append({"person_id": 0,
                          "person_name": "unbekannt"})

for image_path in Path(images_path).glob("*.jpg"):
    # print(image_path.stem)
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    bboxes["images"].append({"image_name": image_path.name,
                             "image_id": img_id,
                             "width": w,
                             "height": h})
    detector.setInputSize((w, h))
    _, faces = detector.detect(image) # 15 Zahlen Output: 4 für bbox, 5*2 für Augen, Nase, Mund, confidence score
    if faces is not None:
        for face in faces: 
            x,y,bw,bh = face[:4]
            cv2.rectangle(image, (int(x), int(y)), (int(x) + int(bw), int(y) + int(bh)), (0, 255, 0), 2)
            right_eye_x, right_eye_y = int(face[4]), int(face[5])
            left_eye_x, left_eye_y = int(face[6]), int(face[7])
            mid_x, mid_y = int((right_eye_x + left_eye_x) / 2), int((right_eye_y + left_eye_y) / 2)
            x_dist, y_dist = right_eye_x - left_eye_x, right_eye_y - left_eye_y
            image = rotate_image(image, (mid_x, mid_y), math.degrees(np.arctan((y_dist/x_dist))))
            print(math.degrees(np.arctan(-(y_dist/x_dist))))
            # cv2.circle(image, (mid_x, mid_y), 3, (0,0,255), -1)     # -1 füllt den Kreis aus
            cv2.line(image, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y), (0,0,255), 3)
            bboxes["annotations"].append({"image_id": img_id,
                                          "annotations_id": ann_id,
                                          "bbox": [int(x), int(y), int(bw), int(bh)],
                                          "person_id": 0,
                                          })
            ann_id = ann_id + 1
    img_id = img_id + 1

    cv2.imshow("YuNet Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# print(bboxes)

with open((images_path / "annotations.json"), "w") as f:
    json.dump(bboxes, f)
    