import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()

    # h, w, c
    h, w, c = frame.shape
    half_w = w // 2

    half_frame = frame[:, :half_w]

    if not ret:
        print("프레임을 받을 수 없습니다.")
        break

    cv2.imshow("camera", half_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()