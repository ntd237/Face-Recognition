import cv2
import os
import pandas as pd

# Đọc hình ảnh từ camera hoặc file video
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set chiều rộng video
cam.set(4, 480)  # Set chiều cao video

# Khởi tạo bộ phân loại Haar Cascade
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Yêu cầu người dùng nhập vào user id
face_id = input('\n Enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count = 0

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1) # Lật hình ảnh để thuận tiện quan sát
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển đổi hình ảnh sang xám
    faces = face_detector.detectMultiScale(gray, 1.3, 5) # Phát hiện khuôn mặt

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) # Vẽ hộp giới hạn quanh khuôn mặt
        count += 1

        # Lưu hình ảnh vào thư mục dataset mới
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w]) # Lưu hình ảnh vào thư mục dataset
        
        cv2.imshow('image', img)  # Hiển thị hình ảnh có phát hiện khuôn mặt


    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows() 