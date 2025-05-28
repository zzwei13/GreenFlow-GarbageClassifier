import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
import time
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import ImageFont, ImageDraw, Image

# 垃圾分類類別
class_dic = {
    0: "cardboard",
    1: "Glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

# 初始化 OLED
serial = i2c(port=1, address=0x3C)  # 根據 OLED 設定調整
oled = ssd1306(serial, width=128, height=64)
font = ImageFont.load_default()

# 分配 TensorRT 引擎的緩衝區
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        dic = {"host_mem": host_mem, "device_mem": device_mem, "shape": engine.get_binding_shape(binding), "dtype": dtype}
        if engine.binding_is_input(binding):
            inputs.append(dic)
        else:
            outputs.append(dic)
    stream = cuda.Stream()
    return inputs, outputs, bindings, stream

# 加載 TensorRT 引擎
def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)

# 預處理影像
def preprocess_frame(frame):
    img = cv2.resize(frame, (256, 256))
    img = np.array(img, dtype=np.float32, order="C") / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.transpose((2, 0, 1))
    return img

# Softmax 函式
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# 執行推論
def do_inference(context, inputs, outputs, bindings, stream):
    [cuda.memcpy_htod_async(inp["device_mem"], inp["host_mem"], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out["host_mem"], out["device_mem"], stream) for out in outputs]
    stream.synchronize()
    return outputs[0]["host_mem"]

# 顯示分類結果在 OLED 上
def display_oled_info(fps, label, confidence):
    image = Image.new("1", (oled.width, oled.height))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
    draw.text((0, 0), f"FPS: {fps:.2f}", font=font, fill=255)
    draw.text((0, 20), f"Label: {label}", font=font, fill=255)
    draw.text((0, 40), f"Conf: {confidence:.2f}", font=font, fill=255)
    oled.display(image)

# 主推論函數
def infer_from_camera(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    # 加載 TensorRT 引擎
    engine = load_engine(trt_runtime, engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Starting inference... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # 預處理影像
        preprocessed = preprocess_frame(frame)
        np.copyto(inputs[0]["host_mem"], preprocessed.ravel())

        # 推論
        start_time = time.time()
        output = do_inference(context, inputs, outputs, bindings, stream)
        infer_time = time.time() - start_time

        # 處理推論結果
        probabilities = softmax(output)
        label_idx = np.argmax(probabilities)
        label = class_dic[label_idx]
        confidence = probabilities[label_idx]

        # 顯示結果在 OLED
        fps = 1.0 / infer_time
        display_oled_info(fps, label, confidence)

        # 在窗口中顯示影像 (可選)
        cv2.putText(frame, f"{label}: {confidence:.2f} (FPS: {fps:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Garbage Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Garbage Classification with OLED Display")
    parser.add_argument("--engine", type=str, required=True, help="Path to the TensorRT engine file")
    args = parser.parse_args()
    infer_from_camera(args.engine)
