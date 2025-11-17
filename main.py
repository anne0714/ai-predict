import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
from contextlib import asynccontextmanager
from model import CNNTransformer 

# ----------------------------------------------------
# 1. 初始化模型和轉換器
# ----------------------------------------------------
MODEL_PATH = "model_weights.pth"
DEVICE = torch.device("cpu") # 部署在 CPU 環境，如果使用 GPU 請改為 "cuda" 

# 定義圖片預處理的轉換，必須和訓練時一致
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(), # 將 PIL Image 轉為 Tensor [1, 28, 28] 且自動將值縮放至 0-1
])

model = None 

# 透過 asynccontextmanager 確保模型只載入一次
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 應用程式啟動時執行 (載入模型)
    print("正在啟動，開始載入模型...")
    global model
    try:
        model = CNNTransformer()
        # 每個請求只佔用一個CPU，這確保了多個並發請求可以平均分配 CPU 核心
        torch.set_num_threads(1)
        # 載入權重。注意：您必須確保 model_weights.pth 檔案存在。
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # 設定為評估模式，關閉 Dropout 等
        print("模型載入成功！")
    except FileNotFoundError:
        print(f"錯誤：模型權重檔案未找到於 {MODEL_PATH}。請確認檔案位置。")
        model = None
    except Exception as e:
        print("模型載入失敗，詳細錯誤資訊如下：")
        # ⚠️ 使用 traceback.print_exc() 輸出完整的錯誤堆疊
        traceback.print_exc() 
        model = None
        
    yield # 應用程式運行中

    # 應用程式關閉時執行
    print("應用程式關閉...")

# 初始化 FastAPI 應用程式
app = FastAPI(
    title="CNN-Transformer 推理服務",
    description="使用 FastAPI 部署 PyTorch 模型",
    version="1.0.0",
    lifespan=lifespan 
)


# ----------------------------------------------------
# 2. 定義 API 端點
# ----------------------------------------------------

# 推理端點：接收圖片並返回預測結果
@app.post("/predict_image/",
          # 讓 Swagger/OpenAPI 介面中顯示更友好的名稱
          summary="對單張圖片執行 AI 分類推論", 
          tags=["AI Inference"])
async def predict_image(file: UploadFile = File(
    ...,
    description="待推論的圖片檔案 (PNG 或 JPG)。圖片將被轉換為 28x28 灰階圖進行預測。")):
    """
    ### 服務使用說明
    接收一個圖片檔案，將其傳送給 CNN-Transformer 模型進行預測。
    
    該模型訓練用於對手寫數字或其他單通道 28x28 圖像進行分類。
    
    **回傳**：預測的數字類別 (0-9)。
    """
    if model is None:
        return JSONResponse(status_code=500, content={"message": "Model not loaded. Please check server logs."})

    try:
        # 1. 讀取圖片檔案
        contents = await file.read()
        # 使用 PIL 從位元組流中打開圖片，並轉為灰階 (L)，這和您 inference.ipynb 中的操作一致
        img = Image.open(BytesIO(contents)).convert("L") 

        # 2. 預處理
        img_tensor = transform(img).unsqueeze(0).to(DEVICE) # [1, 1, 28, 28]

        # 3. 模型推理
        with torch.no_grad():
            output = model(img_tensor)
            # 取得預測結果 (假設是分類問題，輸出 10 個類別的機率)
            _, predicted_class = torch.max(output, 1)

        # 4. 回傳結果
        # item() 將 PyTorch Tensor 轉為標準 Python 數字，以便 JSON 序列化
        return {
            "filename": file.filename, 
            "predicted_class": predicted_class.item(),
            "detail": "Prediction successful."
        }
    
    except Exception as e:
        # 捕獲處理過程中的所有錯誤 (例如圖片格式錯誤)
        return JSONResponse(status_code=500, content={"message": f"Processing failed: {e}"})

# 檢查端點
@app.get("/")
def read_root():
    return {"status": "ok" if model is not None else "error", 
            "message": "API is running!"}