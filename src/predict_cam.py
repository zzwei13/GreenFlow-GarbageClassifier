import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# 定義類別名稱
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 定義模型
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=len(classes)):
        super(GarbageClassifier, self).__init__()
        self.network = models.resnet50(pretrained=False)  # 使用 ResNet50
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)  # 修改輸出層
        
    def forward(self, x):
        return self.network(x)

# 加載模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GarbageClassifier()
model.load_state_dict(torch.load("model_weights.pt", map_location=device))
model = model.to(device)
model.eval()

# 定義影像預處理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet 的標準化參數
])

# 初始化攝像頭
cap = cv2.VideoCapture(0)  # 0 表示默認攝像頭

if not cap.isOpened():
    print("無法開啟攝像頭")
    exit()

print("攝像頭初始化完成，按 'q' 鍵退出程式")

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法獲取影像")
        break

    # 將 OpenCV BGR 格式轉為 PIL 格式
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 預處理影像
    input_tensor = transform(pil_image).unsqueeze(0).to(device)  # 增加 batch 維度
    
    # 模型推論
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    # 在影像上顯示結果
    cv2.putText(frame, f"Class: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Garbage Classifier", frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
