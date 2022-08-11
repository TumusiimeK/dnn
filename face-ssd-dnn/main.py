import cv2
import numpy as np

prototxt_path = "weights/deploy.prototxt.txt"
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

test_image = cv2.imread("faces.jpg")
height, width = test_image.shape[:2]

blob = cv2.dnn.blobFromImage(test_image, 1.0, (350, 350), (104.0, 177.0, 123.0))

model.setInput(blob)
output = np.squeeze(model.forward())

font_scale = 1.0
for i in range(0, output.shape[0]):
    confidence = output[i, 2]
    if confidence > 0.5:
        box = output[i, 3:7]*np.array([width, height, width, height])
        start_x, start_y, end_x, end_y = box.astype(np.int)
        cv2.rectangle(test_image, (start_x, start_y), (end_x, end_y), color=(0,0,255), thickness=2)
        cv2.putText(test_image, f"{confidence*100:.2f}%", (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), 2)

        cv2.imshow("test_image", test_image)
        #cv2.waitkey(0)
        cv2.imwrite("Detected_faces.jpg", test_image)
