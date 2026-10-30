# GreenFlow 綠流 — Smart Garbage Classifier ♻️

以 **ResNet50** 進行垃圾影像分類，並透過 **TensorRT** 部署到 **NVIDIA Jetson Nano**，即時辨識攝影機畫面中的垃圾類別，結果顯示於 OLED 螢幕。

📄 [期末報告簡報（PDF）](docs/GreenFlow_report.pdf)

🏆 **2025 智慧感測聯網創新應用競賽 — 智慧視覺組 佳作**

> 課程期末團隊專案（3 人）

---

## 功能 | Features

- **六類垃圾分類**：cardboard、glass、metal、paper、plastic、trash
- **即時辨識**：讀取 webcam 影像，逐幀推論並顯示類別
- **邊緣部署**：模型轉為 TensorRT engine，在 Jetson Nano 上以 GPU 加速推論，約 **16 FPS**
- **硬體顯示**：透過 I2C 連接 SSD1306 OLED（128×64），顯示 FPS、類別與信心值

---

## 系統流程 | Pipeline

```
Kaggle 垃圾分類資料集 + 自行拍攝的影像
  ──► 資料預處理（Resize 256×256、ImageNet 標準化）
  ──► 微調 ImageNet 預訓練 ResNet50（PyTorch）
  ──► PC 端 webcam 即時測試（predict_cam.py）
  ──► 匯出 ONNX → 轉換為 TensorRT engine
  ──► Jetson Nano 即時推論 + OLED 顯示（transform.py）
```

---

## 模型結果 | Results

- 以 ImageNet 預訓練的 ResNet50 進行遷移學習，替換最後一層為 6 類輸出
- 訓練流程參考 Kaggle 公開筆記本 [[PyTorch] Garbage Classification (~95% accuracy)](https://www.kaggle.com/code/aadhavvignesh/pytorch-garbage-classification-95-accuracy)（by aadhavvignesh），並在此基礎上調整資料切分與超參數
- 訓練 8 個 epoch，驗證集準確率約 **93%**（最佳 94.1%）
- Jetson Nano 實機推論約 **16 FPS**

---

## 專案結構 | Structure

| 檔案 | 說明 |
|---|---|
| `src/split.py` | 將資料集依類別隨機切分為訓練集與驗證集 |
| `src/predict_cam.py` | 載入 PyTorch 權重，以 webcam 即時分類（PC 端） |
| `src/transform.py` | 載入 TensorRT engine，在 Jetson Nano 上即時分類並顯示於 OLED |
| `src/inference_recycle.py` | 物件偵測版本的推論程式（實驗性） |
| `docs/GreenFlow_report.pdf` | 期末報告簡報 |

---

## 技術 | Tech Stack

- **模型**：PyTorch, torchvision（ResNet50）
- **部署**：ONNX, TensorRT, PyCUDA, NVIDIA Jetson Nano
- **影像處理**：OpenCV, Pillow
- **硬體**：SSD1306 OLED（luma.oled）, USB 攝影機

---

## 使用方式 | Usage

### 1. 切分資料集

```bash
python src/split.py
```

### 2. PC 端即時測試

將訓練好的權重 `model_weights.pt` 放在執行目錄下：

```bash
python src/predict_cam.py
```

按 `q` 結束。

### 3. Jetson Nano 部署

```bash
pip install luma.oled luma.core
python src/transform.py --engine <path/to/model.engine>
```

