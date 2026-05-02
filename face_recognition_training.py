from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
import torch
import json
from pathlib import Path
import cv2

images_path = Path("/home/finn/Arduino/ESP32_Cam/images")

with open(images_path / "annotations.json", "r") as file:
    annotation_file = json.load(file)

img_name_dict = {img["image_id"]: img["image_name"] for img in annotation_file["images"]}

image_list = []

for i, ann in enumerate(annotation_file["annotations"]):
    x, y, w, h = ann["bbox"]
    image_name = img_name_dict[ann["image_id"]]
    image_list.append(
        cv2.imread(images_path / image_name)[y:y+h, x:x+w]
    )
    #cv2.imshow("face", image_list[i])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# vortrainiertes efficientnet_b1 nehmen und embeddings der vorletzten layer extrahieren
model = efficientnet_b1(weights = EfficientNet_B1_Weights.IMAGENET1K_V2)
# Ausgabe der layer "flatten" nehmen und in "embeddings" speichern
feature_extractor = create_feature_extractor(model, return_nodes = {"flatten": "embedding"}) 
feature_extractor.eval() # in evaluations modus setzen, da wir efficientnet nicht trainieren sondern nur benutzen

# jetzt müssen alle bilder auf 240x240 Pixel gebracht werden, dazu definition einer Funktion zum Transformieren von Bildern:
mean = [0.485, 0.456, 0.406] # werte aus offizieller Doku: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b1.html#torchvision.models.EfficientNet_B1_Weights
std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.ToPILImage(),          # OpenCV → PIL
    transforms.Resize((240, 240)),    # Bildgröße auf EfficientNet-Input
    transforms.ToTensor(),            # PIL → Tensor [0..1]
    transforms.Normalize(mean, std)
])

# jetzt preprocess auf alle Bilder anwenden, dann den crop ins Modell geben und aus der vorletzten Schicht extrahieren
embeddings = []
for image in image_list:
    prepped_image = preprocess(image)
    prepped_image = prepped_image.unsqueeze(0)
    with torch.no_grad():
        emb = feature_extractor(prepped_image)["embedding"].squeeze(0)
    embeddings.append(emb)
print(len(embeddings[1]))


