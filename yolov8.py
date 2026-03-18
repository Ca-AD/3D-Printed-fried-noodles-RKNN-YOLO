import cv2
import numpy as np
from rknn.api import RKNN

# ==== 配置 ====
RKNN_MODEL = 'best.rknn'          # 你的RKNN模型路径
IMG_PATH = 'test.jpg'             # 测试图片
CONF_THRES = 0.25
NMS_THRES = 0.45
CLASS_NAMES = ['fail_spaghetti']  # data.yaml 中的 names

# ==== YOLOv8后处理 ====
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_threshold):
    """非极大值抑制"""
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = []
        for j in idxs[1:]:
            xx1 = max(boxes[i][0], boxes[j][0])
            yy1 = max(boxes[i][1], boxes[j][1])
            xx2 = min(boxes[i][2], boxes[j][2])
            yy2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            area1 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            area2 = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0
            ious.append(iou)
        ious = np.array(ious)
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def post_process(outputs, conf_thres=0.25, nms_thres=0.45):
    pred = outputs[0].reshape(-1, 6)
    boxes, confs, class_ids = [], [], []
    for p in pred:
        obj_conf = sigmoid(p[4])
        cls_conf = sigmoid(p[5:])
        cls_id = np.argmax(cls_conf)
        conf = obj_conf * cls_conf[cls_id]
        if conf > conf_thres:
            x, y, w, h = p[:4]
            boxes.append([x - w/2, y - h/2, x + w/2, y + h/2])
            confs.append(conf)
            class_ids.append(cls_id)
    if not boxes:
        return [], [], []
    keep = nms(boxes, confs, nms_thres)
    boxes = [boxes[i] for i in keep]
    confs = [confs[i] for i in keep]
    class_ids = [class_ids[i] for i in keep]
    return boxes, confs, class_ids

# ==== 初始化RKNN ====
rknn = RKNN()
print('--> Load RKNN model')
ret = rknn.load_rknn(RKNN_MODEL)
if ret != 0:
    print('load_rknn failed')
    exit(ret)
print('done')

print('--> Init runtime')
ret = rknn.init_runtime(target='rk3588')
if ret != 0:
    print('Init runtime failed')
    exit(ret)
print('done')

# ==== 推理 ====
img = cv2.imread(IMG_PATH)
orig_h, orig_w = img.shape[:2]
input_img = cv2.resize(img, (640, 640))
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_data = np.expand_dims(input_img, 0)

print('--> Inference')
outputs = rknn.inference(inputs=[input_data])
print('done')

# ==== 后处理 ====
boxes, confs, class_ids = post_process(outputs, CONF_THRES, NMS_THRES)

# ==== 绘制检测结果 ====
for (box, conf, cid) in zip(boxes, confs, class_ids):
    x1 = int(box[0] / 640 * orig_w)
    y1 = int(box[1] / 640 * orig_h)
    x2 = int(box[2] / 640 * orig_w)
    y2 = int(box[3] / 640 * orig_h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
    label = f'{CLASS_NAMES[cid]} {conf:.2f}'
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

cv2.imwrite('result.jpg', img)
print('✅ 检测完成，结果已保存为 result.jpg')

rknn.release()
