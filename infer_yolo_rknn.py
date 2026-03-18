import cv2
import numpy as np
from rknn.api import RKNN

def read_image_confidence(model_path, image_path, labels, conf_threshold=0.25):
    """
    读取图片中检测到的目标的置信度
    
    参数:
        model_path: RKNN模型路径
        image_path: 图片路径
        labels: 类别标签列表
        conf_threshold: 置信度阈值
    
    返回:
        detections: 检测结果列表，每个元素为[类别, 置信度, [x1, y1, x2, y2]]
    """
    # 初始化RKNN
    rknn = RKNN()
    
    # 加载模型
    print('[INFO] Loading model...')
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print('[ERROR] Load RKNN model failed!')
        return []
    
    # 初始化运行时
    print("[INFO] Initializing runtime...")
    ret = rknn.init_runtime(target='rk3588')
    if ret != 0:
        print('[ERROR] Init runtime failed!')
        return []
    
    # 读取和预处理图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return []
    
    orig_height, orig_width = img.shape[:2]
    print(f"[INFO] Image size: {orig_width}x{orig_height}")
    
    # 预处理
    img_input = cv2.resize(img, (640, 640))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    img_input /= 255.0
    
    # 推理
    print('[INFO] Running inference...')
    outputs = rknn.inference(inputs=[img_input])
    
    # 处理输出
    detections = process_outputs(outputs, labels, conf_threshold, orig_width, orig_height)
    
    # 释放资源
    rknn.release()
    
    return detections

def process_outputs(outputs, labels, conf_threshold, img_width, img_height):
    """
    处理模型输出，提取检测结果和置信度
    
    返回:
        detections: 检测结果列表，每个元素为字典
    """
    detections = []
    
    print(f"[DEBUG] Number of outputs: {len(outputs)}")
    
    for i, output in enumerate(outputs):
        print(f"[DEBUG] Output {i} shape: {output.shape}")
        
        # 处理不同的输出格式
        if output.shape == (1, 5, 8400):  # YOLOv8格式
            output = output[0]  # (5, 8400)
            output = output.transpose(1, 0)  # (8400, 5)
            
            for j, det in enumerate(output):
                if len(det) == 5:
                    x, y, w, h, conf = det
                    
                    if conf > conf_threshold:
                        # 转换为像素坐标
                        x1 = int((x - w/2) * img_width)
                        y1 = int((y - h/2) * img_height)
                        x2 = int((x + w/2) * img_width)
                        y2 = int((y + h/2) * img_height)
                        
                        # 确保坐标在图像范围内
                        x1 = max(0, min(x1, img_width-1))
                        y1 = max(0, min(y1, img_height-1))
                        x2 = max(0, min(x2, img_width-1))
                        y2 = max(0, min(y2, img_height-1))
                        
                        # 默认类别为0，或者根据你的模型调整
                        cls_id = 0
                        if len(labels) > 1:
                            # 如果有多个类别，这里需要根据模型输出调整
                            cls_id = 0  # 暂时使用0
                        
                        detection = {
                            'class_id': cls_id,
                            'class_name': labels[cls_id] if cls_id < len(labels) else 'unknown',
                            'confidence': float(conf),
                            'bbox': [x1, y1, x2, y2],
                            'bbox_normalized': [float(x), float(y), float(w), float(h)]
                        }
                        detections.append(detection)
                        
                        print(f"[DETECTION] {detection['class_name']}: {detection['confidence']:.3f} "
                              f"at [{x1}, {y1}, {x2}, {y2}]")
        
        # 可以添加其他输出格式的处理
        else:
            print(f"[WARNING] Unsupported output format: {output.shape}")
    
    return detections

def print_detection_summary(detections):
    """打印检测结果摘要"""
    if not detections:
        print("[INFO] No detections found.")
        return
    
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)
    
    # 按类别分组
    class_summary = {}
    for det in detections:
        cls_name = det['class_name']
        if cls_name not in class_summary:
            class_summary[cls_name] = []
        class_summary[cls_name].append(det['confidence'])
    
    # 打印每个类别的统计信息
    for cls_name, confs in class_summary.items():
        avg_conf = np.mean(confs)
        max_conf = np.max(confs)
        min_conf = np.min(confs)
        count = len(confs)
        
        print(f"Class: {cls_name}")
        print(f"  Count: {count}")
        print(f"  Average Confidence: {avg_conf:.3f}")
        print(f"  Max Confidence: {max_conf:.3f}")
        print(f"  Min Confidence: {min_conf:.3f}")
        print(f"  All Confidences: {[f'{c:.3f}' for c in confs]}")
        print()
    
    # 总体统计
    all_confs = [det['confidence'] for det in detections]
    print(f"Overall - Total detections: {len(detections)}")
    print(f"Overall - Average confidence: {np.mean(all_confs):.3f}")
    print(f"Overall - Highest confidence: {np.max(all_confs):.3f}")
    print("="*50)

def save_detections_to_file(detections, output_file):
    """将检测结果保存到文件"""
    with open(output_file, 'w') as f:
        f.write("class_name,confidence,x1,y1,x2,y2\n")
        for det in detections:
            bbox = det['bbox']
            f.write(f"{det['class_name']},{det['confidence']:.4f},"
                   f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
    print(f"[INFO] Detections saved to {output_file}")

# 使用示例
if __name__ == '__main__':
    # 配置参数
    MODEL_PATH = 'best.rknn'
    IMAGE_PATH = 'test.jpg'
    LABELS = ['spaghetti', 'normal', 'other']  # 根据你的模型调整
    CONF_THRESH = 0.25
    OUTPUT_FILE = 'detections.csv'
    
    # 读取图片置信度
    detections = read_image_confidence(MODEL_PATH, IMAGE_PATH, LABELS, CONF_THRESH)
    
    # 打印摘要
    print_detection_summary(detections)
    
    # 保存到文件
    if detections:
        save_detections_to_file(detections, OUTPUT_FILE)
    
    # 可选：绘制结果
    if detections:
        img = cv2.imread(IMAGE_PATH)
        for det in detections:
            bbox = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # 绘制边界框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # 绘制标签
            cv2.putText(img, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存结果图像
        result_path = 'confidence_result.jpg'
        cv2.imwrite(result_path, img)
        print(f"[INFO] Result image saved to {result_path}")
