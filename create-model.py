import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Specify the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load image
image = cv2.imread("image.jpg")

# Preprocess image
height, width, channels = image.shape
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Initialize lists for detected objects
class_ids = []
confidences = []
boxes = []

# Process the output layers
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Add detected objects to lists
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Apply non-maximum suppression to remove redundant overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Generate markdown output for detected objects
output = ""
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box
    label = str(class_ids[i])
    confidence = confidences[i]
    output += f"- Object: {label}, Confidence: {confidence:.2f}, Location: ({x}, {y}), Width: {w}, Height: {h}\n"

print(output)
