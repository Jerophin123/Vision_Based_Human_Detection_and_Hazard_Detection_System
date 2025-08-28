import cv2

for i in range(5):  # test indices 0..4
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        cap = cv2.VideoCapture(i, backend)
        ok, _ = cap.read()
        cap.release()
        print(("MSMF" if backend==cv2.CAP_MSMF else "DSHOW"), i, ok)
