"""
EVALUATE_TV3.py
Script đánh giá toàn diện cho Task TV3 (Evaluation, Quality Rejection, Open-set & Grad-CAM)
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from freshlens_ai.inference.cnn_predict import FreshLensPredictor

def run_evaluation():
    print("="*60)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ TV3 - FRESHLENS V2")
    print("="*60)

    model_path = "models/cnn_efficientnet_b0/best.pt"
    openset_path = "models/checkpoints/openset_gate.json"
    data_dir = "data/test" # Đường dẫn tập test
    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("error_gallery", exist_ok=True)

    # Khởi tạo predictor
    predictor = FreshLensPredictor(
        model_path=model_path,
        openset_gate_path=openset_path if os.path.exists(openset_path) else None
    )

    y_true = []
    y_pred = []
    latencies = []
    rejected_quality_count = 0
    rejected_openset_count = 0

    classes = FreshLensPredictor.JOINT_CLASSES

    for class_idx, class_name in enumerate(classes):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            continue

        for img_name in os.listdir(class_folder):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(class_folder, img_name)
            res = predictor.predict(img_path, use_tta=False, check_quality=True)

            latencies.append(res['latency_ms'])

            if not res['is_supported']:
                if res['rejection_type'] == 'quality_rejection':
                    rejected_quality_count += 1
                else:
                    rejected_openset_count += 1
                continue

            y_true.append(class_name)
            y_pred.append(res['joint_class'])

            # Lưu vào Error Gallery nếu đoán sai
            if res['joint_class'] != class_name:
                err_path = os.path.join("error_gallery", f"GT_{class_name}_PRED_{res['joint_class']}_{img_name}")
                Image.open(img_path).convert('RGB').save(err_path)

    print("\n📊 KẾT QUẢ ĐÁNH GIÁ (EVALUATION METRICS):")
    print(f"- Tổng số mẫu xử lý thành công: {len(y_true)}")
    print(f"- Số mẫu bị Rejection (Chất lượng kém): {rejected_quality_count}")
    print(f"- Số mẫu bị Rejection (Open-set / Ngoài phạm vi): {rejected_openset_count}")
    if len(latencies) > 0:
        print(f"- Latency trung bình mỗi ảnh: {np.mean(latencies):.2f} ms")

    if len(y_true) > 0:
        # Lấy động đúng danh sách các lớp xuất hiện thực tế
        labels_present = sorted(list(set(y_true) | set(y_pred)))
        
        # 1. In Báo cáo dạng bảng ra Terminal
        report_text = classification_report(
            y_true, 
            y_pred, 
            labels=labels_present,
            target_names=labels_present, 
            zero_division=0
        )
        print("\n--- BÁO CÁO CHI TIẾT THEO TỪNG LỚP (CLASSIFICATION REPORT) ---")
        print(report_text)

        # 2. Tạo dict lưu vào file JSON
        report_dict = classification_report(
            y_true, 
            y_pred, 
            labels=labels_present,
            target_names=labels_present, 
            output_dict=True, 
            zero_division=0
        )
        
        print("--- TỔNG QUAN (SUMMARY) ---")
        print(f"Accuracy: {report_dict['accuracy']:.4f}")
        print(f"Macro F1-Score: {report_dict['macro avg']['f1-score']:.4f}")

        # Lưu báo cáo vào file JSON
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Đã xuất báo cáo chi tiết vào thư mục: {output_dir}/metrics.json")
        print(f"✅ Đã xuất các mẫu dự đoán sai vào thư mục: error_gallery/")

if __name__ == "__main__":
    run_evaluation()