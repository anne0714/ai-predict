import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import glob
from model import CNNTransformer

# -----------------------------------------------------------------
# 1. 配置與參數
# -----------------------------------------------------------------
MODEL_PATH = "model_weights.pth"
TEST_DIR = "test"  # 測試圖片所在的資料夾名稱 (假設您已建立)
OUTPUT_CSV = "result.csv"
BATCH_SIZE = 64    # 批次大小，可根據記憶體調整
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚠️ 轉換器必須與您的訓練和 FastAPI 服務中的定義完全一致
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(), # → [1, 28, 28]
])

# -----------------------------------------------------------------
# 2. 自定義 Dataset 處理圖片
# -----------------------------------------------------------------
class TestImageDataset(Dataset):
    """用於批量載入測試圖片的自定義 Dataset"""
    def __init__(self, img_dir, transform=None):
        # 使用 glob 獲取所有 .png 或 .jpg 圖片的路徑
        self.img_paths = glob.glob(os.path.join(img_dir, '*.png')) + \
                         glob.glob(os.path.join(img_dir, '*.jpg'))
        self.transform = transform
        
        # 提取檔案名稱作為 ID
        self.file_names = [os.path.basename(p) for p in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("L") # 轉為灰階
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.file_names[idx]

# -----------------------------------------------------------------
# 3. 推論主程式
# -----------------------------------------------------------------
def batch_predict():
    # 載入模型
    try:
        model = CNNTransformer()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"模型 {MODEL_PATH} 載入成功，運行於 {DEVICE}。")
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return

    # 準備資料集
    test_dataset = TestImageDataset(TEST_DIR, transform=transform)
    if len(test_dataset) == 0:
        print(f"⚠️ 在 {TEST_DIR} 資料夾中未找到任何圖片 (.png 或 .jpg)。請檢查路徑。")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"🔎 總共找到 {len(test_dataset)} 張圖片，開始批量推論...")

    results = []

    # 開始推論
    with torch.no_grad():
        for images, file_names in test_loader:
            # 將圖片移動到正確的設備 (CPU/GPU)
            images = images.to(DEVICE)
            
            # 執行模型前向傳播
            outputs = model(images)
            
            # 取得預測類別
            _, predicted_classes = torch.max(outputs.data, 1)
            
            # 儲存結果
            for file_name, prediction in zip(file_names, predicted_classes.cpu().numpy()):
                results.append({
                    "filename": file_name,
                    "prediction": prediction.item() # .item() 轉為標準 Python 數字
                })
    
    # 輸出結果到 CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    
    print(f"批量推論完成！結果已儲存至 {OUTPUT_CSV}")

if __name__ == "__main__":
    batch_predict()