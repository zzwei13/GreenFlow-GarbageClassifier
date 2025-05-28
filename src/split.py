import os
import random
import shutil

def split_dataset(
    source_dir="Garbage classification",
    output_dir="dataset",
    train_ratio=0.7
):
    """
    依照指定比例 (train_ratio) 將 source_dir 內的 6 個類別資料夾，
    分成 train 與 val，並且將影像複製到 output_dir。
    """

    # 定義類別名 (可按實際資料夾名稱增減)
    class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    
    # 建立輸出資料夾 (dataset/train, dataset/val)，以及各自的類別子資料夾
    for phase in ["train", "val"]:
        phase_dir = os.path.join(output_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)
        for cname in class_names:
            os.makedirs(os.path.join(phase_dir, cname), exist_ok=True)

    # 對每個類別進行資料切割
    for cname in class_names:
        class_folder = os.path.join(source_dir, cname)
        if not os.path.exists(class_folder):
            print(f"警告：資料夾不存在 {class_folder}")
            continue

        # 取得所有圖片檔案列表
        files = [f for f in os.listdir(class_folder) 
                 if os.path.isfile(os.path.join(class_folder, f))]
        # 隨機打亂
        random.shuffle(files)

        # 計算 train / val 的分割點
        train_count = int(len(files) * train_ratio)

        # 分配檔案
        train_files = files[:train_count]
        val_files   = files[train_count:]
        
        # 複製檔案到 train 資料夾
        for f in train_files:
            src = os.path.join(class_folder, f)
            dst = os.path.join(output_dir, "train", cname, f)
            shutil.copy2(src, dst)
        
        # 複製檔案到 val 資料夾
        for f in val_files:
            src = os.path.join(class_folder, f)
            dst = os.path.join(output_dir, "val", cname, f)
            shutil.copy2(src, dst)

        print(f"{cname} 類別：共 {len(files)} 張影像，"
              f"訓練集 {len(train_files)}、驗證集 {len(val_files)}")

if __name__ == "__main__":
    # 您可視需求修改 source_dir 與 output_dir
    split_dataset(
        source_dir="Garbage classification\\Garbage classification",
        output_dir="dataset",
        train_ratio=0.7
    )
