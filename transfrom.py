import os
import tensorflow as tf
from PIL import Image

# --- 配置 ---
dataset_path = "./datasets/cifar10"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

train_img_path = os.path.join(images_path, "train")
val_img_path = os.path.join(images_path, "val")
# test_img_path = os.path.join(images_path, "test")  # 可选，如果需要测试集

train_label_path = os.path.join(labels_path, "train")
val_label_path = os.path.join(labels_path, "val")
# test_label_path = os.path.join(labels_path, "test") # 可选


# ... （之前的 create_yolo_labels 函数不变）
def create_yolo_labels(image_height, image_width, bounding_boxes, class_ids):
    """Creates YOLO label strings from bounding box and class information."""
    labels = []
    for bbox, class_id in zip(bounding_boxes, class_ids):
        # CIFAR-10 doesn't have bounding boxes, so we use the whole image
        x_center = 0.5
        y_center = 0.5
        width = 1.0
        height = 1.0
        label = f"{class_id} {x_center} {y_center} {width} {height}\n"
        labels.append(label)
    return labels


def process_cifar10_data(
    x_data, y_data, img_output_path, label_output_path, split="train"
):
    """处理 CIFAR-10 数据并保存图片和标签到指定目录"""

    os.makedirs(img_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)

    for i, (image, label) in enumerate(zip(x_data, y_data)):
        image_pil = Image.fromarray(image)
        image_height, image_width = image_pil.size

        labels = create_yolo_labels(image_height, image_width, [0], label)

        # 保存图片
        image_filename = os.path.join(img_output_path, f"{i}.png")
        image_pil.save(image_filename)

        # 保存标签
        label_filename = os.path.join(label_output_path, f"{i}.txt")
        with open(label_filename, "w") as f:
            f.writelines(labels)

        if i % 1000 == 0:
            print(f"已处理 {split} 分割的 {i} 张图片")


# --- 加载并处理 CIFAR-10 数据集 ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# --- 处理训练集 ---
process_cifar10_data(x_train, y_train, train_img_path, train_label_path, split="train")

# --- 处理验证集 ---
process_cifar10_data(x_test, y_test, val_img_path, val_label_path, split="val")

# --- 处理测试集 (可选)---
# process_cifar10_data(x_test, y_test, test_img_path, test_label_path, split="test")

print("CIFAR-10 转换到 YOLO 格式完成。")
