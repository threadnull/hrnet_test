import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import cv2
import os

from util import decode_heatmap
from model import HRNet

# 하이퍼파라미터
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_KEYPOINTS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# KEYPOINTS = ["center", "tip", "min", "max"]

def inference(model, frame_np, device, input_size=(256, 256)):
    """
    frame_np: OpenCV로 읽은 BGR 이미지 (Numpy array)
    """
    model.eval()

    # 원본 크기
    h, w = frame_np.shape[:2]

    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    # nparray -> tensor
    frame_tensor = F.to_tensor(frame_rgb)
    # 정규화
    frame_tensor = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(frame_tensor)
    # 리사이즈
    frame_input = F.resize(frame_tensor, input_size)
    # 배치차원추가 [C, H, W] -> [1, C, H, W]
    frame_input = frame_input.unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        outputs = model(frame_input)
        
        # Heatmap -> 좌표 변환 (64x64 기준 좌표)
        pred_coords = decode_heatmap(outputs)  # [1, K, 2]

    # 64x64 -> 256x256
    pred_coords = pred_coords * 4.0
    
    # 256x256 기준 좌표를 원본 이미지 크기로 다시 변환
    scale_x = w / input_size[1]
    scale_y = h / input_size[0]
    
    pred_coords[0, :, 0] *= scale_x
    pred_coords[0, :, 1] *= scale_y

    # 좌표반환
    return pred_coords[0]

if __name__ == "__main__":
    try:
        checkpoint = "models/hrnet/hrnet_w32.pth"
        state_dict = torch.load(checkpoint, map_location=DEVICE)
        model = HRNet(num_keypoints=NUM_KEYPOINTS).to(DEVICE)
        model.load_state_dict(state_dict)
        print(f"{os.path.basename(checkpoint)} 모델 불러오기 완료")

    except:
        print("Error: 모델 불러오기 실패")
        exit()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("Error: 카메라 열기 실패")
        exit()

    # info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)])

    print(f"{width} x {height} {codec}")

    try:
        while True:
            ret, frame_np = cap.read()

            if not ret:
                print("Error: 프레임이 없습니다")
                break

            # h, w, c
            h, w, c = frame_np.shape
            half_w = w//2

            l_frame = frame_np[:, :half_w]

            # 추론
            keypoints = inference(model, l_frame, DEVICE)

            # 시각화
            keypoint = keypoints.cpu().numpy()

            if keypoint.ndim == 3:
                keypoint = keypoint[0]

            for pt in keypoint:
                x, y = int(pt[0]), int(pt[1])
                
                if 0 <= x < half_w and 0 <= y < h:
                    # cv2.circle(img, (x, y), 반지름, 색상, 두께)
                    cv2.circle(l_frame, (x, y), 5, (0, 255, 0), -1) # 녹색

            l_frame = cv2.resize(l_frame, (640, 480))
            cv2.imshow("infer", l_frame)

            if cv2.waitKey(1) == ord("q"):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()