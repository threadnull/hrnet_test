import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.io import read_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import timm
import kornia
import numpy as np
from tqdm import tqdm
import os
import json


# imagenet 평균, 표준편차
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# SIGMA
SIGMA = 1.5

# 모델
class GaugeHRNet(nn.Module):
    def __init__(self, num_keypoints=4, pretrained=True, stride_idx=1):
        super(GaugeHRNet, self).__init__()
        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=pretrained,
            features_only=True
        )
        self.stride_idx = stride_idx
        # Stride 4 (64x64) 특징맵 채널 수 가져오기
        in_channels = self.backbone.feature_info.channels()[self.stride_idx]

        self.final_layer = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

    def forward(self, x):
        # 특징맵 리스트 추출
        x = self.backbone(x)[self.stride_idx]
        # Stride 4 (64x64)
        x = self.final_layer(x)
        return x

# 데이터셋
class GaugeDataset(Dataset):
    def __init__(self, root_dir, ann_file, input_size=(256, 256), is_train=True):
        self.root_dir = root_dir
        with open(ann_file, "r") as f:
            self.coco = json.load(f)
        self.image_map = {img["id"]: img for img in self.coco["images"]}
        self.annotaions = self.coco["annotations"]
        self.input_size = input_size
        self.is_train = is_train

        # 정규화
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # 증강
        self.augment = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ]) if is_train else nn.Identity()
        

    def __len__(self):
        return len(self.annotaions)

    def __getitem__(self, index):
        annotaion = self.annotaions[index]
        image_info = self.image_map[annotaion["image_id"]]
        image_path = os.path.join(self.root_dir, image_info["file_name"])

        # Tensor C H W, 리사이즈, 0~255
        image = read_image(image_path)
        image = F.resize(image, self.input_size)
        image = image.float() / 255.0

        # 증강
        image = self.augment(image)
        # 정규화
        image = self.normalize(image)
        
        # 좌표 스케일링
        # 원본 이미지 크기 정보 가져오기
        w = image_info['width']
        h = image_info['height']

        scale_x = self.input_size[1] / w
        scale_y = self.input_size[0] / h

        keypoints = torch.tensor(annotaion["keypoints"]).view(-1, 3).float()
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        return image, keypoints[:, :2]

# 히트맵
def generate_heatmap(keypoints, map_size, sigma=SIGMA):
    """
    [학습용] 좌표 -> 히트맵
    """
    B, N, _ = keypoints.shape
    H, W = map_size
    device = keypoints.device

    # 표준편차
    std = torch.tensor([sigma, sigma], device=device).repeat(B, N, 1)

    # Kornia 히트맵 렌더링
    heatmaps = kornia.geometry.subpix.render_gaussian2d(
        mean=keypoints,
        std=std,
        size=(H, W),
        normalized_coordinates=False # 픽셀 좌표계 사용
    )
    return heatmaps

def decode_heatmap(heatmaps):
    """
    [추론용] 히트맵 -> 좌표
    """
    # Soft Argmax
    # temperature가 높을수록 argmax에 가까워지고, 낮을수록 평균에 가까워짐
    coords = kornia.geometry.subpix.spatial_soft_argmax2d(
        input=heatmaps,
        temperature=torch.tensor(1.0, device=heatmaps.device),
        normalized_coordinates=False
    )
    return coords

# 학습
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, keypoints in tqdm(dataloader, desc="학습중 "):
        images = images.to(device)
        keypoints = keypoints.to(device)

        # hrnet target 1/4
        with torch.no_grad():
            target_keypoints = keypoints / 4.0
            targets = generate_heatmap(target_keypoints, (64, 64), sigma=SIGMA)

            targets = targets * 1000.0

        # 전파
        optimizer.zero_grad()
        outputs = model(images)

        # loss
        loss = criterion(outputs, targets)

        # 역전파
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# 검증
def val(model, dataloader, criterion, device, epoch_idx=0):
    model.eval()
    total_loss = 0
    for i, (images, keypoints) in tqdm(enumerate(dataloader), "검증중 "):
        images = images.to(device)
        keypoints = keypoints.to(device)

        target_keypoints = keypoints / 4.0
        targets = generate_heatmap(target_keypoints, (64, 64), sigma=SIGMA)

        # [중요] 학습 때와 동일하게 스케일링 적용해야 Loss 비교 가능
        targets = targets * 1000.0

        # 전파
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # 첫 번째 배치의 결과만 출력 확인
        if i == 0:
            # 좌표 복원 시에는 Scaling된 히트맵도 위치는 동일하므로 그대로 사용 가능
            pred_coords = decode_heatmap(outputs)
            pred_coords = pred_coords * 4.0
                
    avg_loss = total_loss / len(dataloader)
    return avg_loss
if __name__ == "__main__":
    
    # 설정
    TRAIN_IMG = "coco_dataset/train/images"
    TRAIN_ANN = "coco_dataset/train/labels/person_keypoints_Train.json"
    VAL_IMG = "coco_dataset/val/images"
    VAL_ANN = "coco_dataset/val/labels/person_keypoints_Validation.json"
    SAVE_PATH = "models/hrnet/hrnet.pth"
    NUM_KEYPOINTS = 4
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCH = 100
    print(DEVICE)

    # 모델
    model = GaugeHRNet(num_keypoints=NUM_KEYPOINTS).to(DEVICE)

    # 데이터셋 로드
    if os.path.exists(TRAIN_ANN) and os.path.exists(VAL_ANN):
        train_dataset = GaugeDataset(TRAIN_IMG, TRAIN_ANN, is_train=True)
        val_dataset = GaugeDataset(VAL_IMG, VAL_ANN, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        print(f"학습 시작 (Device: {DEVICE})")
        print("-" * 50)

        best_val_loss = float("inf")

        for epoch in range(EPOCH):
            train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
            val_loss = val(model, val_loader, criterion, DEVICE)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{EPOCH} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # 성능 개선 시 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"모델 저장: {val_loss:.6f}")

            print("-" * 50)
        print("학습 완료 및 모델 저장됨.")

    else:
        print("데이터셋 파일이 없습니다. 경로를 확인해주세요.")