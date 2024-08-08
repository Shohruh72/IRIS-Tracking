import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10, refine_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

color = (0, 255, 255)
white_color = (255, 255, 255)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Camera parameters
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('demo.mp4', fourcc, fps, (width * 2, height))


def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist / total_distance
    if ratio <= 0.42:
        return "right", ratio
    elif 0.42 < ratio <= 0.57:
        return "center", ratio
    else:
        return "left", ratio


def draw_table(image, left_iris_pos, left_ratio, left_radius, right_iris_pos, right_ratio, right_radius):
    rows, cols, _ = image.shape
    table_start_y = rows - 150
    cv2.rectangle(image, (0, table_start_y), (cols, rows), (50, 50, 50), -1)  # Draw table background

    # Draw table headers
    cv2.putText(image, "EYES", (30, table_start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white_color, 2)
    cv2.putText(image, "POSITION", (200, table_start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white_color, 2)
    cv2.putText(image, "RATIO", (350, table_start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white_color, 2)
    cv2.putText(image, "RADIUS", (500, table_start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white_color, 2)

    # Draw table values
    cv2.putText(image, "LEFT", (30, table_start_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, left_iris_pos, (200, table_start_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, f"{left_ratio:.2f}", (350, table_start_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, f"{left_radius:.2f}", (500, table_start_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(image, "RIGHT", (30, table_start_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, right_iris_pos, (200, table_start_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, f"{right_ratio:.2f}", (350, table_start_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, f"{right_radius:.2f}", (500, table_start_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        mesh_points = np.array([np.multiply([p.x, p.y], [image.shape[1], image.shape[0]]).astype(int) for p in
                                results.multi_face_landmarks[0].landmark])

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)

        cv2.circle(image, center_left, int(l_radius), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(image, center_right, int(r_radius), (255, 255, 0), 1, cv2.LINE_AA)

        right_iris_pos, right_ratio = iris_position(center_right, mesh_points[33], mesh_points[133])
        left_iris_pos, left_ratio = iris_position(center_left, mesh_points[362], mesh_points[263])

        pad = np.zeros_like(image)

        cv2.circle(pad, center_left, int(l_radius), color, 3, cv2.LINE_AA)
        cv2.circle(pad, center_right, int(r_radius), color, 3, cv2.LINE_AA)

        draw_table(pad, left_iris_pos, left_ratio, l_radius, right_iris_pos, right_ratio, r_radius)
    else:
        pad = np.zeros_like(image)

    combined_image = np.hstack((image, pad))

    cv2.imshow('Iris Tracking', combined_image)
    out.write(combined_image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
