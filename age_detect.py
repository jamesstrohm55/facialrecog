import cv2
import numpy as np

# Paths to the model files
face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"

# Mean values and age labels used by the pre-trained model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load the pre-trained models
face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

def detect_faces(net, frame, conf_threshold=0.7):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)
    
    return frame, face_boxes

def predict_age(face, net):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    net.setInput(blob)
    age_preds = net.forward()
    print("Raw age predictions:", age_preds[0])  # <-- Add this line
    age = age_list[age_preds[0].argmax()]
    
    return age


def process_image(image_path):
    # Load the image
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Image not found at {image_path}")
        return

    # Detect faces
    frame, face_boxes = detect_faces(face_net, frame)

    # Predict age for each face
    for (x1, y1, x2, y2) in face_boxes:
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (227, 227))

        age = predict_age(face, age_net)
        cv2.putText(frame, f"Age: {age}", (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the result
    cv2.imshow("Age Estimation", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    image_path = "IMG_8193.JPEG"  # Change to your image path
    process_image(image_path)
