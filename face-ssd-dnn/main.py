# Import the required libraries...
import cv2
import numpy as np

# Create variables to hold the model with the pre-trained weights...
prototxt_path = "weights/deploy.prototxt.txt"
model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Load the Model...
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Read the Test Image...
test_image = cv2.imread("faces.jpg")
# Capture the height and width of the test image
height, width = test_image.shape[:2]

# Preprocess the image (resize & perform mean subtraction)
blob = cv2.dnn.blobFromImage(test_image, 1.0, (350, 350), (104.0, 177.0, 123.0))

# Input the image into the Neural-Network
model.setInput(blob)
# Perform feed forward to get detected faces
output = np.squeeze(model.forward())

font_scale = 1.0
for i in range(0, output.shape[0]):
    confidence = output[i, 2]    # Get the confidence
     # If confidence is above 50%, draw the surrounding box
    if confidence > 0.5:
        box = output[i, 3:7]*np.array([width, height, width, height])
        # Covert to integers
        start_x, start_y, end_x, end_y = box.astype(np.int)
        # Draw the rectangle surrounding the face
        cv2.rectangle(test_image, (start_x, start_y), (end_x, end_y), color=(0,0,255), thickness=2)
        # Add text that shows the confidence of the model
        cv2.putText(test_image, f"{confidence*100:.2f}%", (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), 2)

         # show the image
        cv2.imshow("test_image", test_image)
        # save the detected faces image with rectangles
        cv2.imwrite("Detected_faces.jpg", test_image)
