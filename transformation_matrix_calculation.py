import cv2
import numpy as np

def select_points(frame):
    src_points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            src_points.append([x, y])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Select 4 Points', frame)
            if len(src_points) == 4:
                print("4 points selected. Press any key to continue...")
    cv2.imshow('Select 4 Points', frame)
    cv2.setMouseCallback('Select 4 Points', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return src_points

def main():
    cap = cv2.VideoCapture("http://10.255.32.36:8080/?action=stream")
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        return
    selected_points = select_points(frame)
    if len(selected_points) == 4:
        selected_points = np.float32(selected_points)
        width, height = 800, 600
        dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv2.getPerspectiveTransform(selected_points, dst_points)
        np.save('transformation_matrix.npy', M)
        print("Transformation matrix saved.")

if __name__ == "__main__":
    main()
