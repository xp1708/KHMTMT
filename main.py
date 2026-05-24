import cv2
import mediapipe as mp
import numpy as np
import time
import math
import platform

# =========================
# TUY CHON CHUNG
# =========================
CAMERA_INDEX = 0
USE_PHONE_DETECTION = True

# =========================
# NGUONG MAT / MIENG
# =========================
EAR_THRESHOLD = 0.20            # Eye Aspect Ratio - duoi nguong nay coi la mat nham
MAR_THRESHOLD = 0.50            # Mouth Aspect Ratio - tren nguong nay coi la dang ngap
EYE_CLOSED_SECONDS = 1.3        # Thoi gian mat nham lien tuc de canh bao
YAWN_SECONDS = 1.0              # Thoi gian ngap lien tuc de canh bao
DISTRACT_SECONDS = 4.5          # Thoi gian nhin lech lien tuc de canh bao
NO_FACE_SECONDS = 2.0           # Thoi gian khong thay mat de canh bao
ALERT_COOLDOWN_SECONDS = 2.0    # Khoang nghi giua 2 lan phat am thanh canh bao

# =========================
# TOI UU HEAD POSE
# =========================
CALIBRATION_SECONDS = 2.0       # Thoi gian hieu chinh pose chuan ban dau
HEAD_POSE_ALPHA = 0.25          # He so EMA filter (cang nho cang muot)
MAX_ANGLE_JUMP = 20.0           # Gioi han nhay goc giua 2 frame de loc nhieu
MIN_FACE_BOX_SIZE = 120         # Kich thuoc mat toi thieu de tinh head pose

YAW_ENTER_THRESHOLD = 20.0      # Goc yaw de vao trang thai lech
YAW_EXIT_THRESHOLD = 15.0       # Goc yaw de thoat trang thai lech (hysteresis)
PITCH_ENTER_THRESHOLD = 20.0    # Goc pitch de vao trang thai guc dau
PITCH_EXIT_THRESHOLD = 12.0     # Goc pitch de thoat trang thai guc dau (hysteresis)

# Neu tren may ban cui dau lam pitch duong thi de = 1
# Neu cui dau lam pitch am thi doi thanh -1
HEAD_DROP_SIGN = 1

# =========================
# PHAT HIEN DIEN THOAI
# =========================
PHONE_MODEL_PATH = "yolo11n.onnx"
PHONE_CONF_THRESHOLD = 0.45             # Confidence toi thieu cua detection
PHONE_USE_SECONDS = 1.5                 # Thoi gian cam dien thoai lien tuc de canh bao
PHONE_DETECT_EVERY_N_FRAMES = 4         # Chay YOLO moi N frame de tiet kiem CPU
PHONE_IOU_WITH_FACE_THRESHOLD = 0.02    # IoU toi thieu giua box phone va box mat
PHONE_NEAR_FACE_EXPAND_PX = 180         # Mo rong vung mat de bat dien thoai gan mat

# =========================
# KIEM TRA ANH SANG
# =========================
# Hai metric:
#   - too_dark: mean brightness qua thap (anh toi thuc su)
#   - too_noisy: mean thap + std thap = camera dang boost gain, anh nhieu phang
# Logitech C922 co auto-exposure nen can ca 2 metric. Tinh chinh theo setup thuc te.
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 60     # Nguong mean grayscale (0-255)
LOW_LIGHT_NOISE_THRESHOLD = 15          # Nguong std grayscale khi camera boost gain
LOW_LIGHT_SECONDS = 1.5                 # Thoi gian anh sang yeu lien tuc de canh bao

# =========================
# MEDIAPIPE LANDMARK
# =========================
# 6 diem moi mat theo chuan EAR (Soukupova & Cech 2016)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# 3 cap diem doc mieng + 1 cap diem ngang mieng de tinh MAR
MOUTH_TOP_BOTTOM = [(13, 14), (81, 178), (311, 402)]
MOUTH_LEFT_RIGHT = (78, 308)

# 6 landmark dung cho solvePnP de tinh huong dau
HEAD_POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}


def play_alert():
    """Phat am thanh canh bao. Tren Pi/Linux dung ASCII Bell, tren Windows dung winsound."""
    try:
        if platform.system().lower().startswith("win"):
            import winsound
            winsound.Beep(1500, 250)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def check_lighting(frame, brightness_threshold=60, noise_threshold=15):
    """
    Kiem tra dieu kien anh sang cua frame.

    Tra ve (is_low_light, mean, std):
      - is_low_light: True neu anh qua toi HOAC qua nhieu
      - mean: gia tri brightness trung binh (0-255)
      - std: do lech chuan brightness, dung de phat hien camera boost gain

    Ly do dung 2 metric: C922 co auto-exposure se tu boost gain khi thieu sang,
    lam mean van cao nhung anh nhieu, MediaPipe se fail. Dung mean + std de bat
    ca 2 truong hop.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    std = gray.std()
    too_dark = mean < brightness_threshold
    too_noisy = (mean < 90) and (std < noise_threshold)
    return too_dark or too_noisy, mean, std


def dist(p1, p2):
    """Khoang cach Euclidean giua 2 diem 2D."""
    return float(np.linalg.norm(np.array(p1, dtype=np.float64) - np.array(p2, dtype=np.float64)))


def eye_aspect_ratio(eye_points):
    """
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Mat mo: EAR ~ 0.25-0.30. Mat nham: EAR ~ 0.
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = dist(p2, p6)
    vertical_2 = dist(p3, p5)
    horizontal = dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(face_points):
    """MAR = trung_binh(khoang_cach_doc) / khoang_cach_ngang_mieng."""
    w = dist(face_points[MOUTH_LEFT_RIGHT[0]], face_points[MOUTH_LEFT_RIGHT[1]])
    if w == 0:
        return 0.0
    v = sum(dist(face_points[a], face_points[b]) for a, b in MOUTH_TOP_BOTTOM) / len(MOUTH_TOP_BOTTOM)
    return v / w


def normalize_angle(angle):
    """Chuan hoa goc ve khoang (-180, 180] de tranh angle wrapping."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return float(angle)


def smooth_value(prev_val, new_val, alpha=0.25):
    """EMA filter: S_t = alpha * x_t + (1-alpha) * S_{t-1}."""
    if prev_val is None:
        return float(new_val)
    return float(alpha * new_val + (1 - alpha) * prev_val)


def clamp_angle_jump(prev_val, new_val, max_jump=20.0):
    """Gioi han bien do thay doi goc giua 2 frame de loai outlier truoc khi EMA."""
    if prev_val is None:
        return float(new_val)
    delta = normalize_angle(new_val - prev_val)
    delta = max(-max_jump, min(max_jump, delta))
    return float(prev_val + delta)


def rotation_matrix_to_euler_angles(R):
    """Chuyen rotation matrix 3x3 sang Euler angles (pitch, yaw, roll) theo do."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])   # pitch
        y = math.atan2(-R[2, 0], sy)       # yaw
        z = math.atan2(R[1, 0], R[0, 0])   # roll
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees(np.array([x, y, z], dtype=np.float64))


def get_face_box(face_points):
    """Tinh bounding box bao toan bo landmark khuon mat."""
    xs = [p[0] for p in face_points]
    ys = [p[1] for p in face_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max, (x_max - x_min), (y_max - y_min)


def box_iou(boxA, boxB):
    """Intersection over Union giua 2 bounding box."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def expand_box(box, expand_px, frame_w, frame_h):
    """Mo rong box ra 4 phia expand_px pixel, clamp theo kich thuoc frame."""
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1 - expand_px))
    y1 = max(0, int(y1 - expand_px))
    x2 = min(frame_w - 1, int(x2 + expand_px))
    y2 = min(frame_h - 1, int(y2 + expand_px))
    return (x1, y1, x2, y2)


def center_in_box(point, box):
    """Kiem tra mot diem co nam trong bounding box khong."""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def estimate_head_pose(face_points, frame_width, frame_height, prev_rvec=None, prev_tvec=None):
    """
    Uoc luong huong dau bang solvePnP voi 6 landmark khuon mat.
    Neu co prev_rvec/tvec se dung lam initial guess de hoi tu nhanh hon.
    """
    image_points = np.array(
        [
            face_points[HEAD_POSE_LANDMARKS["nose_tip"]],
            face_points[HEAD_POSE_LANDMARKS["chin"]],
            face_points[HEAD_POSE_LANDMARKS["left_eye_outer"]],
            face_points[HEAD_POSE_LANDMARKS["right_eye_outer"]],
            face_points[HEAD_POSE_LANDMARKS["mouth_left"]],
            face_points[HEAD_POSE_LANDMARKS["mouth_right"]],
        ],
        dtype=np.float64,
    )

    # Toa do 3D chuan hoa cua khuon mat trung binh (don vi mm)
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )

    # Camera matrix gia dinh tieu cu = frame width (xap xi webcam thong thuong)
    focal_length = frame_width
    center = (frame_width / 2.0, frame_height / 2.0)

    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    if prev_rvec is not None and prev_tvec is not None:
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            prev_rvec,
            prev_tvec,
            True,
            cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_mat)

    pitch = normalize_angle(pitch)
    yaw = normalize_angle(yaw)
    roll = normalize_angle(roll)

    # Project 1 diem o truoc mui de ve vector huong dau
    nose_end_point2d, _ = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)], dtype=np.float64),
        rotation_vec,
        translation_vec,
        camera_matrix,
        dist_coeffs,
    )

    nose_tip = tuple(map(int, image_points[0]))
    nose_direction = tuple(map(int, nose_end_point2d[0][0]))

    return {
        "pitch": float(pitch),
        "yaw": float(yaw),
        "roll": float(roll),
        "nose_tip": nose_tip,
        "nose_direction": nose_direction,
        "rvec": rotation_vec,
        "tvec": translation_vec,
    }


def put_status_text(frame, status, color=(0, 255, 0)):
    """Ve banner trang thai o goc tren ben trai."""
    cv2.rectangle(frame, (10, 10), (760, 95), (0, 0, 0), -1)
    cv2.putText(frame, f"Trang thai: {status}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "Nhan Q de thoat", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)


def median_of_tuples(data):
    """Tinh median tren tung chieu cua tap tuple (dung cho calibration)."""
    arr = np.array(data, dtype=np.float64)
    return np.median(arr[:, 0]), np.median(arr[:, 1]), np.median(arr[:, 2])


def load_phone_model():
    """Load YOLO11n ONNX model va parse class names tu metadata."""
    if not USE_PHONE_DETECTION:
        return None, {}
    try:
        import onnxruntime as ort
        import ast
        session = ort.InferenceSession(
            PHONE_MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        meta = session.get_modelmeta().custom_metadata_map
        names = {}
        if "names" in meta:
            names = ast.literal_eval(meta["names"])
        return session, names
    except Exception as e:
        print(f"[WARN] Khong tai duoc model phone: {e}")
        return None, {}


def detect_phone(model, frame_bgr):
    """Chay YOLO inference, loc detection cua class 'cell phone' (COCO id=67)."""
    detections = []
    if model is None:
        return detections
    try:
        PHONE_CLASS_ID = 67
        # Preprocess: BGR -> RGB -> resize 640x640 -> normalize -> NCHW
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (640, 640))
        img_input = img_resized.astype(np.float32) / 255.0
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)

        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: img_input})

        predictions = outputs[0][0].T
        orig_h, orig_w = frame_bgr.shape[:2]
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        for pred in predictions:
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])
            if class_id != PHONE_CLASS_ID or conf < PHONE_CONF_THRESHOLD:
                continue
            # Convert cx,cy,w,h -> x1,y1,x2,y2 va scale ve kich thuoc frame goc
            x1 = max(0, int((cx - w / 2) * scale_x))
            y1 = max(0, int((cy - h / 2) * scale_y))
            x2 = min(orig_w - 1, int((cx + w / 2) * scale_x))
            y2 = min(orig_h - 1, int((cy + h / 2) * scale_y))
            detections.append({
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "label": "cell phone",
            })
    except Exception as e:
        print(f"[WARN] Loi detect phone: {e}")
    return detections


def is_phone_usage(phone_dets, face_box, frame_w, frame_h):
    """
    Chi coi la dang dung dien thoai neu:
    - box phone giao voi face box mot chut
    HOAC
    - tam phone nam trong vung face box mo rong
    """
    if not phone_dets or face_box is None:
        return False, None

    expanded_face = expand_box(face_box, PHONE_NEAR_FACE_EXPAND_PX, frame_w, frame_h)

    for det in phone_dets:
        phone_box = det["box"]
        iou = box_iou(phone_box, face_box)
        cx = (phone_box[0] + phone_box[2]) // 2
        cy = (phone_box[1] + phone_box[3]) // 2

        near_face = center_in_box((cx, cy), expanded_face)
        overlap_face = iou >= PHONE_IOU_WITH_FACE_THRESHOLD

        if near_face or overlap_face:
            return True, det

    return False, None


def main():
    mp_face_mesh = mp.solutions.face_mesh
    phone_model, _ = load_phone_model()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Khong mo duoc webcam. Kiem tra CAMERA_INDEX hoac quyen camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ===== TIMER STATE =====
    eye_closed_since = None
    yawn_since = None
    distracted_since = None
    no_face_since = None
    phone_since = None
    low_light_since = None
    last_alert_time = 0.0

    # ===== HEAD POSE STATE =====
    prev_rvec = None
    prev_tvec = None
    smooth_pitch = None
    smooth_yaw = None
    smooth_roll = None

    # ===== CALIBRATION STATE =====
    base_pitch = None
    base_yaw = None
    base_roll = None
    calibration_start = None
    calibration_samples = []

    # ===== HYSTERESIS STATE =====
    distracted_state = False
    head_drop_state = False

    # ===== FRAME / FPS =====
    frame_count = 0
    cached_phone_dets = []
    fps_start = time.time()
    fps_counter = 0
    fps_display = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            now = time.time()  # FIX: Gan timestamp som de cac block phia sau dung duoc

            # ===== KIEM TRA ANH SANG =====
            # Tinh ngay sau khi co frame de quyet dinh co tin tuong cac detection sau khong
            is_low_light, light_mean, light_std = check_lighting(
                frame,
                LOW_LIGHT_BRIGHTNESS_THRESHOLD,
                LOW_LIGHT_NOISE_THRESHOLD
            )
            if is_low_light:
                if low_light_since is None:
                    low_light_since = now
            else:
                low_light_since = None

            # ===== FPS COUNTER =====
            frame_count += 1
            fps_counter += 1
            elapsed_fps = now - fps_start
            if elapsed_fps >= 1.0:
                fps_display = fps_counter / elapsed_fps
                fps_counter = 0
                fps_start = now

            # ===== PHONE DETECTION (chay moi N frame de tiet kiem CPU) =====
            if USE_PHONE_DETECTION and phone_model is not None:
                if frame_count % PHONE_DETECT_EVERY_N_FRAMES == 0:
                    cached_phone_dets = detect_phone(phone_model, frame)

            # ===== FACE MESH =====
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status = "Tap trung"
            status_color = (0, 255, 0)
            alert_messages = []

            ear = 0.0
            mar = 0.0
            pitch_rel = 0.0
            yaw_rel = 0.0
            roll_rel = 0.0
            current_face_box = None

            if results.multi_face_landmarks:
                no_face_since = None
                face_landmarks = results.multi_face_landmarks[0]

                # Quy doi tat ca landmark sang toa do pixel
                face_points = []
                for lm in face_landmarks.landmark:
                    x = lm.x * frame_w
                    y = lm.y * frame_h
                    face_points.append((x, y))

                # ===== TINH EAR / MAR =====
                left_eye = [face_points[i] for i in LEFT_EYE_IDX]
                right_eye = [face_points[i] for i in RIGHT_EYE_IDX]

                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)
                ear = (ear_left + ear_right) / 2.0
                mar = mouth_aspect_ratio(face_points)

                x_min, y_min, x_max, y_max, box_w, box_h = get_face_box(face_points)
                face_size = min(box_w, box_h)
                current_face_box = (int(x_min), int(y_min), int(x_max), int(y_max))

                pose_valid = False
                pose = None

                # ===== HEAD POSE (chi tinh khi mat du lon) =====
                if face_size >= MIN_FACE_BOX_SIZE:
                    pose = estimate_head_pose(face_points, frame_w, frame_h, prev_rvec, prev_tvec)
                    if pose is not None:
                        pose_valid = True
                        prev_rvec = pose["rvec"]
                        prev_tvec = pose["tvec"]

                        raw_pitch = pose["pitch"]
                        raw_yaw = pose["yaw"]
                        raw_roll = pose["roll"]

                        # Clamp truoc khi smooth de outlier khong "o nhiem" lich su EMA
                        raw_pitch = clamp_angle_jump(smooth_pitch, raw_pitch, MAX_ANGLE_JUMP)
                        raw_yaw = clamp_angle_jump(smooth_yaw, raw_yaw, MAX_ANGLE_JUMP)
                        raw_roll = clamp_angle_jump(smooth_roll, raw_roll, MAX_ANGLE_JUMP)

                        smooth_pitch = smooth_value(smooth_pitch, raw_pitch, HEAD_POSE_ALPHA)
                        smooth_yaw = smooth_value(smooth_yaw, raw_yaw, HEAD_POSE_ALPHA)
                        smooth_roll = smooth_value(smooth_roll, raw_roll, HEAD_POSE_ALPHA)

                        # ===== CALIBRATION POSE CHUAN =====
                        if base_pitch is None or base_yaw is None or base_roll is None:
                            if calibration_start is None:
                                calibration_start = now
                                calibration_samples = []

                            calibration_samples.append((smooth_pitch, smooth_yaw, smooth_roll))
                            elapsed = now - calibration_start

                            status = "Dang hieu chinh pose... giu dau thang"
                            status_color = (255, 255, 0)

                            if elapsed >= CALIBRATION_SECONDS and len(calibration_samples) >= 15:
                                base_pitch, base_yaw, base_roll = median_of_tuples(calibration_samples)

                        # Tinh goc tuong doi so voi baseline ca nhan
                        if base_pitch is not None:
                            pitch_rel = normalize_angle(smooth_pitch - base_pitch)
                            yaw_rel = normalize_angle(smooth_yaw - base_yaw)
                            roll_rel = normalize_angle(smooth_roll - base_roll)

                        cv2.line(frame, pose["nose_tip"], pose["nose_direction"], (255, 255, 0), 2)
                    else:
                        prev_rvec = None
                        prev_tvec = None

                # Ve cac landmark quan trong de debug
                key_ids = LEFT_EYE_IDX + RIGHT_EYE_IDX + [p for pair in MOUTH_TOP_BOTTOM for p in pair] + list(MOUTH_LEFT_RIGHT)
                key_ids = sorted(set(key_ids + list(HEAD_POSE_LANDMARKS.values())))

                for idx in key_ids:
                    x, y = face_points[idx]
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)

                # ===== CAP NHAT TIMER =====
                eye_closed = ear < EAR_THRESHOLD
                yawning = mar > MAR_THRESHOLD

                eye_closed_since = now if eye_closed and eye_closed_since is None else (eye_closed_since if eye_closed else None)
                yawn_since = now if yawning and yawn_since is None else (yawn_since if yawning else None)

                distracted_now = False
                head_drop_now = False

                # ===== HYSTERESIS CHO HEAD POSE =====
                if pose_valid and base_pitch is not None:
                    yaw_limit = YAW_EXIT_THRESHOLD if distracted_state else YAW_ENTER_THRESHOLD
                    distracted_now = abs(yaw_rel) > yaw_limit

                    pitch_limit = PITCH_EXIT_THRESHOLD if head_drop_state else PITCH_ENTER_THRESHOLD
                    head_drop_now = (HEAD_DROP_SIGN * pitch_rel) > pitch_limit

                    distracted_state = distracted_now
                    head_drop_state = head_drop_now

                distracted_since = now if distracted_now and distracted_since is None else (distracted_since if distracted_now else None)

                # ===== KIEM TRA NGUONG THOI GIAN -> THEM CANH BAO =====
                if eye_closed_since is not None and (now - eye_closed_since) >= EYE_CLOSED_SECONDS:
                    alert_messages.append("BUON NGU: nham mat qua lau")

                if yawn_since is not None and (now - yawn_since) >= YAWN_SECONDS:
                    alert_messages.append("BUON NGU: phat hien ngap")

                if head_drop_now and eye_closed_since is not None and (now - eye_closed_since) >= EYE_CLOSED_SECONDS:
                    alert_messages.append("BUON NGU: guc dau va nham mat")

                if distracted_since is not None and (now - distracted_since) >= DISTRACT_SECONDS:
                    alert_messages.append("MAT TAP TRUNG: nhin lech huong")

                # Ve face box va cac chi so debug
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (120, 120, 120), 1)

                cv2.putText(frame, f"EAR: {ear:.3f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.3f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Pitch(rel): {pitch_rel:.1f}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Yaw(rel): {yaw_rel:.1f}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Roll(rel): {roll_rel:.1f}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if base_pitch is None:
                    cv2.putText(
                        frame,
                        "Dang hieu chinh pose: nhin thang vao camera",
                        (20, 280),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
            else:
                # ===== KHONG THAY MAT - RESET STATE =====
                eye_closed_since = None
                yawn_since = None
                distracted_since = None
                prev_rvec = None
                prev_tvec = None
                smooth_pitch = None
                smooth_yaw = None
                smooth_roll = None
                distracted_state = False
                head_drop_state = False

                if no_face_since is None:
                    no_face_since = now

                if (now - no_face_since) >= NO_FACE_SECONDS:
                    status = "Mat tap trung (Khong thay khuon mat)"
                    status_color = (0, 165, 255)
                    alert_messages.append("MAT TAP TRUNG: khong phat hien khuon mat")

            # ===== PHONE USAGE LOGIC =====
            phone_in_use = False
            phone_det_used = None

            if USE_PHONE_DETECTION and current_face_box is not None and len(cached_phone_dets) > 0:
                phone_in_use, phone_det_used = is_phone_usage(
                    cached_phone_dets, current_face_box, frame_w, frame_h
                )

            if phone_in_use:
                if phone_since is None:
                    phone_since = now
            else:
                phone_since = None

            if phone_since is not None and (now - phone_since) >= PHONE_USE_SECONDS:
                alert_messages.append("MAT TAP TRUNG: dang su dung dien thoai")

            # Ve box dien thoai (mau magenta neu chi detect duoc, do neu dang dung)
            for det in cached_phone_dets:
                x1, y1, x2, y2 = det["box"]
                conf = det["conf"]

                color = (255, 0, 255)
                if phone_det_used is not None and det["box"] == phone_det_used["box"]:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"Phone {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # ===== XEP TRANG THAI THEO UU TIEN =====
            has_drowsy = any(msg.startswith("BUON NGU") for msg in alert_messages)
            has_phone = any("dien thoai" in msg for msg in alert_messages)
            has_distracted = any(msg.startswith("MAT TAP TRUNG") for msg in alert_messages)

            if has_phone:
                status = "Dang dung dien thoai"
                status_color = (255, 0, 255)
            elif has_drowsy and has_distracted:
                status = "Buon ngu + Mat tap trung"
                status_color = (0, 0, 255)
            elif has_drowsy:
                status = "Buon ngu"
                status_color = (0, 0, 255)
            elif has_distracted:
                status = "Mat tap trung"
                status_color = (0, 165, 255)

            # ===== FIX: OVERRIDE STATE KHI ANH SANG YEU =====
            # Phai chay TRUOC put_status_text de status hien thi dung ngay frame dau tien.
            # Khi anh sang yeu, MediaPipe khong dang tin -> bo qua moi canh bao khac
            # de tranh false positive (vd: "khong phat hien khuon mat" do toi qua).
            low_light_active = (
                low_light_since is not None and (now - low_light_since) >= LOW_LIGHT_SECONDS
            )
            if low_light_active:
                alert_messages.clear()
                status = "Anh sang yeu - khong the giam sat"
                status_color = (0, 165, 255)

            # ===== RENDER STATUS VA ALERT MESSAGES =====
            put_status_text(frame, status, status_color)
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (frame_w - 160, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            for i, msg in enumerate(alert_messages[:4]):
                if msg.startswith("BUON NGU"):
                    color = (0, 0, 255)
                elif "dien thoai" in msg:
                    color = (255, 0, 255)
                else:
                    color = (0, 165, 255)

                cv2.putText(
                    frame,
                    msg,
                    (20, 330 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            # ===== OVERLAY CANH BAO ANH SANG YEU O DAY DUOI =====
            if low_light_active:
                cv2.rectangle(frame, (0, frame_h - 60), (frame_w, frame_h), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"CANH BAO: Anh sang yeu (mean={light_mean:.0f}, std={light_std:.0f})",
                    (20, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

            # ===== PHAT AM THANH CANH BAO (co cooldown) =====
            if alert_messages and (now - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
                play_alert()
                last_alert_time = now

            cv2.imshow("He thong phat hien buon ngu, mat tap trung va dien thoai", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()