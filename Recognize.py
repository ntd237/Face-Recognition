import cv2
import numpy as np
import pandas as pd
import os

# Đường dẫn tới file mô hình nhận diện khuôn mặt đã huấn luyện
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Đường dẫn tới tệp XML của Haar Cascade
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Đọc dữ liệu từ dataset_model.csv
dataset_path = 'dataset.csv'
df = pd.read_csv(dataset_path)

# Khởi tạo cửa sổ hiển thị
font = cv2.FONT_HERSHEY_SIMPLEX

# Kích thước tối thiểu của khuôn mặt
minW = 0.1 * 640
minH = 0.1 * 480

# Khởi tạo camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set chiều rộng video
cam.set(4, 480)  # set chiều cao video

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # lật hình ảnh

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # chuyển hình ảnh sang xám

    # phát hiện khuôn mặt
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    # vẽ hộp giới hạn quanh khuôn mặt và nhận diện khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Dự đoán id của khuôn mặt từ ảnh xám
        id_predicted, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        
        # Kiểm tra độ chính xác của dự đoán
        if confidence < 100:
            id_predicted = df.loc[df['id'] == id_predicted, 'name'].values[0]
            confidence_text = "  {0}%".format(round(100 - confidence))
        else:
            id_predicted = "unknown"
            confidence_text = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id_predicted), (x + 5, y - 5), font, 1, (255, 255, 255), 2)  # hiển thị tên
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)  # hiển thị độ chính xác

    cv2.imshow('camera', img)

    # Nhấn 'ESC' để thoát
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
