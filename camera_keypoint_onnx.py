import cv2
import numpy as np
import onnxruntime as ort

# 하이퍼파라미터
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = (256, 256)
ONNX_MODEL_PATH = "models/onnx/hrnet_w32.onnx"

def preprocess(frame, input_size):
    """
    이미지 전처리: BGR -> RGB -> Resize -> Normalize -> Transpose(HWC->CHW)
    """
    # 1. BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Resize
    frame_resized = cv2.resize(frame_rgb, (input_size[1], input_size[0]))
    
    # 3. Normalize (0~1 Scaling & Standardize)
    frame_norm = frame_resized.astype(np.float32) / 255.0
    frame_norm = (frame_norm - IMAGENET_MEAN) / IMAGENET_STD
    
    # 4. HWC -> CHW (Channel First)
    frame_transposed = frame_norm.transpose(2, 0, 1)
    
    # 5. Add Batch Dimension: (1, C, H, W)
    input_tensor = frame_transposed[np.newaxis, ...].astype(np.float32)
    
    return input_tensor

def soft_argmax_numpy(heatmaps):
    """
    Kornia의 spatial_soft_argmax2d를 NumPy로 구현
    heatmaps shape: (1, K, H, W)
    """
    b, c, h, w = heatmaps.shape
    
    # 1. Flatten spatial dimensions (1, K, H*W)
    heatmaps_flat = heatmaps.reshape(b, c, -1)
    
    # 2. Softmax
    # max를 빼주는 것은 exp 계산 시 오버플로우 방지 (Numerical Stability)
    heatmaps_flat = heatmaps_flat - np.max(heatmaps_flat, axis=2, keepdims=True)
    heatmaps_prob = np.exp(heatmaps_flat)
    heatmaps_prob = heatmaps_prob / np.sum(heatmaps_prob, axis=2, keepdims=True)
    
    # 3. Reshape back to (1, K, H, W)
    heatmaps_prob = heatmaps_prob.reshape(b, c, h, w)
    
    # 4. Generate meshgrid
    grid_x = np.arange(w, dtype=np.float32)
    grid_y = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(grid_x, grid_y) # xv: (H, W), yv: (H, W)
    
    # 5. Expectation (Weighted Sum)
    # coords_x: sum(prob * x_pos)
    coords_x = np.sum(heatmaps_prob * xv[np.newaxis, np.newaxis, :, :], axis=(2, 3))
    coords_y = np.sum(heatmaps_prob * yv[np.newaxis, np.newaxis, :, :], axis=(2, 3))
    
    # Stack to (1, K, 2) -> (x, y)
    coords = np.stack([coords_x, coords_y], axis=-1)
    return coords

def main():
    # 1. ONNX Runtime 세션 초기화
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        print(f"{ONNX_MODEL_PATH} 로드 완료. Device: {ort.get_device()}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    # 입력/출력 이름 가져오기
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Inference Start (Press 'q' to exit)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 원본 크기
            h_orig, w_orig = frame.shape[:2]
            
            # 왼쪽 절반만 사용 (기존 로직 유지)
            half_w = w_orig // 2
            l_frame = frame[:, :half_w]
            h_crop, w_crop = l_frame.shape[:2]

            # 2. 전처리
            input_tensor = preprocess(l_frame, INPUT_SIZE)

            # 3. 추론 (run 메서드 사용)
            # outputs는 리스트 형태로 반환되므로 [0]번 인덱스 사용
            outputs = ort_session.run([output_name], {input_name: input_tensor})[0]

            # 4. 후처리 (Heatmap -> Coords)
            pred_coords = soft_argmax_numpy(outputs) # (1, K, 2)

            # 좌표 복원 로직
            # 64x64(출력) -> 256x256(입력) -> Original Crop Size
            # 모델 출력(64)에서 입력(256)으로 갈 때 4배 확대
            pred_coords = pred_coords * 4.0 
            
            # 입력(256)에서 원본 Crop(w_crop, h_crop)으로 확대
            scale_x = w_crop / INPUT_SIZE[1]
            scale_y = h_crop / INPUT_SIZE[0]
            
            pred_coords[0, :, 0] *= scale_x
            pred_coords[0, :, 1] *= scale_y

            # 5. 시각화
            keypoints = pred_coords[0] # (K, 2)
            
            for pt in keypoints:
                x, y = int(pt[0]), int(pt[1])
                # 범위 체크
                if 0 <= x < w_crop and 0 <= y < h_crop:
                    cv2.circle(l_frame, (x, y), 5, (0, 0, 255), -1) # 빨간색

            # 결과 보기
            vis_frame = cv2.resize(l_frame, (640, 480))
            cv2.imshow("ONNX Inference", vis_frame)

            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()