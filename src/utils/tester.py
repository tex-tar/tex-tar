import os
import json
import cv2
import matplotlib.pyplot as plt
# 1) Paths
json_path  = "/data/textar_outputs/jsons/generate_bbox.json"
image_root = "/data/textar_inputs"
image_name = "003.png"

# 2) Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# 3) Get the bboxes for 001.png
bboxes = data.get(image_name, [])
if not bboxes:
    print(f"No boxes found for {image_name}")
    exit()

# 4) Load the image
img_path = os.path.join(image_root, image_name)
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

# 5) Iterate, print & draw
for entry in bboxes:
    x0, y0, x1, y1 = entry["bb_dim"]
    ocr_text = entry["bb_ids"][0]["ocrv"]
    print(f"Box: ({x0}, {y0}) → ({x1}, {y1})   Text: '{ocr_text}'")
  
    # draw a green rectangle
    cv2.rectangle(img, (x0, y0), (x1, y1), color=(0,255,0), thickness=1)
    # optionally overlay the OCR text:

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out_path = os.path.join(image_root, "001_boxes.png")

# Save
plt.imsave(out_path, img_rgb)

# 6) Show the result
plt.imsave(f"{image_name} boxes.png", img_rgb)
