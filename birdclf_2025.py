import numpy as np
import seaborn
import matplotlib.pyplot as plt
import soundfile as sf
import glob
import os
import librosa
import librosa.display
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

# data merge
essential_sound_train_df = sound_train_df[["primary_label", "filename", "rating"]]
essential_sound_taxonomy_df = sound_taxonomy_df[[ "primary_label","scientific_name", "common_name", "class_name"]]

merge_df = pd.merge(essential_sound_train_df, essential_sound_taxonomy_df, how = "inner")



#전체 고유한 클래스 ID 추출
class_list = sorted(merge_df["primary_label"].unique())

# 클래스 ID → 인덱스 매핑 딕셔너리
label_to_index = {label: idx for idx, label in enumerate(class_list)}

# 인덱스 → 클래스 ID (역변환용)
index_to_label = {idx: label for label, idx in label_to_index.items()}
merge_df["label_idx"] = merge_df["primary_label"].astype(str).map(label_to_index)



#### 모델 ####

class BirdImageDataset(Dataset):
    def __init__(self, df, mel_img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.mel_img_dir = mel_img_dir  # mel 이미지 저장 경로
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # .png 이미지 경로 생성
        label = str(row["primary_label"])
        filename = os.path.basename(row["filename"]).replace(".ogg", ".png")
        img_path = os.path.join(self.mel_img_dir, label, filename)

        # 이미지 로드
        image = Image.open(img_path).convert("RGB")

        # 변환 적용 (ToTensor 등)
        if self.transform:
            image = self.transform(image)

        label_idx = int(row["label_idx"])
        return image, label_idx


transform = transforms.Compose([
    transforms.Resize((64, 128)),  # 원하는 크기로 조절
    transforms.ToTensor(),
])

dataset = BirdImageDataset(merge_df, mel_img_dir=r"/content/drive/MyDrive/data/mel_images", transform=transform)

from torch.utils.data import random_split

# 전체 데이터셋 크기
dataset_size = len(dataset)

# Train/Test 비율
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size

# 데이터셋 나누기
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 각각 DataLoader 만들기
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)



for images, labels in train_loader:
    print(images.shape)
    print(labels)
    break  # 한 배치만 확인



import torch.nn as nn
import torch.nn.functional as F

# 모델 정의
class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()

        # Convolution Blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Adaptive Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((4, 4))

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_classes = merge_df["label_idx"].nunique()

model = DeeperCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print(optimizer)

# 학습 루프
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

    # -----------------------------
    # Validation 파트 추가
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
