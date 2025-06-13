# Age Detection with OpenCV DNN

This project uses OpenCV's deep learning (DNN) module to detect human faces in an image and estimate the age range of each detected face using a pre-trained Caffe model.

## 📸 Example Output

The program draws bounding boxes around detected faces and displays estimated age ranges like:

'(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'


---

## 🧠 Models Used

- **Face Detection**:  
  - `opencv_face_detector.pbtxt`  
  - `opencv_face_detector_uint8.pb`
  
- **Age Estimation**:  
  - `age_deploy.prototxt`  
  - `age_net.caffemodel`

All models are publicly available and pre-trained.

---

## 📁 File Structure

facialrecog/
│
├── age_detect.py # Main Python script
├── age_net.caffemodel # Age estimation model (binary)
├── age_deploy.prototxt # Age estimation model config
├── opencv_face_detector.pbtxt # Face detection config
├── opencv_face_detector_uint8.pb # Face detection model
└── README.md


---

## ✅ Requirements

- Python 3.8+
- OpenCV 4.x
- NumPy

Install with:

```bash
pip install opencv-python numpy

⚠️ Limitations
Predicts from fixed age ranges:
(0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)

Model may be inaccurate under poor lighting, occlusions, or non-frontal faces.

Trained on limited data (Adience dataset).

📌 Future Improvements
Use more robust models like:

UTKFace-trained CNN

MediaPipe or OpenVINO for faster and more accurate results

Add gender prediction

Add live webcam support
