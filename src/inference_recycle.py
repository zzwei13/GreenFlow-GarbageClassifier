"""
inference.py Infernce the model on nano with video & 
show the FPS & class name &number of each class in frame on OLED
"""

""" OLED套件
pip install luma.oled
pip install luma.core
"""
import sys

sys.path.append("/usr/lib/python3.6/dist-packages/tensorrt")

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
from pathlib import Path
import cv2
from PIL import Image
import argparse  # <- yaml 已移除
import time
import pafy
from collections import Counter

from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import ImageFont, ImageDraw, Image

# 初始化 OLED
serial = i2c(port=1, address=0x3C)  # 根據 OLED 設定調整
oled = ssd1306(serial, width=128, height=64)  # OLED 顯示尺寸
font = ImageFont.load_default()


# 顯示 FPS 和每類別計數在 OLED 上
def display_oled_info(device, fps, class_counts, label):
    image = Image.new("1", (device.width, device.height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, device.width, device.height), outline=0, fill=0)
    draw.text((0, 0), f"FPS: {fps:.2f}", font=font, fill=255)

    y_offset = 10
    for class_id, count in class_counts.items():
        draw.text((0, y_offset), f"{label[class_id]}: {count}", font=font, fill=255)
        y_offset += 10

    device.display(image)


def non_maximum_suppression_fast(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    pick = []

    x1 = boxes[:, 0].astype("float")
    y1 = boxes[:, 1].astype("float")
    x2 = boxes[:, 2].astype("float")
    y2 = boxes[:, 3].astype("float")

    bound_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    sort_index = np.argsort(y2)

    while sort_index.shape[0] > 0:
        last = sort_index.shape[0] - 1
        i = sort_index[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[sort_index[:last]])
        yy1 = np.maximum(y1[i], y1[sort_index[:last]])
        xx2 = np.minimum(x2[i], x2[sort_index[:last]])
        yy2 = np.minimum(y2[i], y2[sort_index[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / bound_area[sort_index[:last]]

        sort_index = np.delete(
            sort_index, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    return pick


def load_engine(trt_runtime, plan_path):
    engine = trt_runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
    return engine


def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        dic = {
            "host_mem": host_mem,
            "device_mem": device_mem,
            "shape": engine.get_binding_shape(binding),
            "dtype": dtype,
        }
        if engine.binding_is_input(binding):
            inputs.append(dic)
        else:
            outputs.append(dic)

    stream = cuda.Stream()
    return inputs, outputs, bindings, stream


def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed)


def do_inference(context, pics_1, inputs, outputs, bindings, stream, model_output_shape):
    start = time.perf_counter()
    load_images_to_buffer(pics_1, inputs[0]["host_mem"])

    [cuda.memcpy_htod_async(inp["device_mem"], inp["host_mem"], stream) for inp in inputs]
    context.execute(batch_size=1, bindings=bindings)
    [cuda.memcpy_dtoh_async(out["host_mem"], out["device_mem"], stream) for out in outputs]
    stream.synchronize()

    out = outputs[0]["host_mem"].reshape((outputs[0]["shape"]))
    return out, time.perf_counter() - start


def draw_detect(img, x1, y1, x2, y2, conf, class_id, label, color_palette):
    color = color_palette[class_id]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        f"{label[class_id]} {conf:0.3}",
        (x1 - 10, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def show_detect(img, preds, iou_threshold, conf_threshold, class_label, color_palette):
    boxes = []
    scores = []
    class_ids = []

    max_conf = np.max(preds[0, 4:, :], axis=0)
    idx_list = np.where(max_conf > conf_threshold)[0]

    for pred_idx in idx_list:
        pred = preds[0, :, pred_idx]
        conf = pred[4:]

        box = [
            pred[0] - 0.5 * pred[2],
            pred[1] - 0.5 * pred[3],
            pred[0] + 0.5 * pred[2],
            pred[1] + 0.5 * pred[3],
        ]
        boxes.append(box)

        label = np.argmax(conf)
        scores.append(max_conf[pred_idx])
        class_ids.append(label)

    boxes = np.array(boxes)
    result_boxes = non_maximum_suppression_fast(boxes, overlapThresh=iou_threshold)

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]

        draw_detect(
            img,
            round(box[0]),
            round(box[1]),
            round(box[2]),
            round(box[3]),
            scores[index],
            class_ids[index],
            class_label,
            color_palette,
        )

    return [
        {"boxes": boxes[i], "class_id": class_ids[i], "conf": scores[i]}
        for i in result_boxes
    ]


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs=1, type=str, help="model path")
    parser.add_argument("--source", nargs=1, type=str, help="inference target")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--data", nargs=1, type=str, help="dataset path (unused)")
    parser.add_argument("--show", action="store_true", help="show detect result")

    opt = parser.parse_args()
    return opt


def main(opt):
    print(opt)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    engine_path = opt["weights"][0]
    source = opt["source"][0]
    iou_threshold = opt["iou_thres"]
    conf_threshold = opt["conf_thres"]
    show = opt["show"]

    ### 修改區：yaml邏輯移除後，直接手動指定 label ###
    # 原本：
    #   yaml_path = opt["data"][0]
    #   with open(yaml_path, "r") as stream:
    #       data = yaml.load(stream)
    #   label = data["names"]
    #   color_palette = np.random.uniform(0, 255, size=(len(label), 3))

    # 新增: 自訂 label & color_palette
    label = ["class0", "class1", "class2"]
    color_palette = np.random.uniform(0, 255, size=(len(label), 3))
    ### 修改區結束 ###

    engine = load_engine(trt_runtime, engine_path)

    video_inferences(source, engine, iou_threshold, conf_threshold, label, color_palette, show)


def video_inferences(video_path, engine, iou_threshold, conf_threshold, label, color_palette, show):
    inputs, outputs, bindings, stream = allocate_buffers(engine, 1)
    context = engine.create_execution_context()

    WIDTH = inputs[0]["shape"][2]
    HEIGHT = inputs[0]["shape"][3]
    model_output_shape = outputs[0]["shape"]

    video_info = "video"
    if "youtube.com" in video_path:
        video_info = pafy.new(video_path)
        video_path = video_info.getbest(preftype="mp4").url
    elif len(video_path.split(".")) == 1:
        video_info = "webcam"
        video_path = int(video_path)

    print(f"Inference with : {video_info}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("VideoCapture Error")
        return

    font = ImageFont.load_default()
    black_image = Image.new("1", (oled.width, oled.height), "black")
    oled.display(black_image)
    current_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = np.array(im, dtype=np.float32, order="C")
        im = im.transpose((2, 0, 1))
        im /= 255

        out, infer_time = do_inference(context, im, inputs, outputs, bindings, stream, model_output_shape)
        total_time = time.perf_counter() - start_time

        detect_results = show_detect(frame, out, iou_threshold, conf_threshold, label, color_palette)

        fps = 1 / (time.perf_counter() - start_time)
        class_counts = Counter([result["class_id"] for result in detect_results])

        image = Image.new("1", (oled.width, oled.height), "black")
        draw = ImageDraw.Draw(image)

        draw.text((0, 0), f"FPS: {fps:.2f}", font=font, fill=255)
        y_offset = 12
        for class_id, count in class_counts.items():
            draw.text((0, y_offset), f"{label[class_id]}: {count}", font=font, fill=255)
            y_offset += 12

        if image != current_image:
            oled.display(image)
            current_image = image

        time.sleep(0.05)

        if show:
            cv2.imshow("img", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    oled.display(black_image)
    print("推理完成，資源已釋放。")


if __name__ == "__main__":
    opt = parse_opt()
    main(vars(opt))
