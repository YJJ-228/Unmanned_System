# Just a homework for Unmanned System

## 1. Convention Part
**convention**文件夹内使用随机森林进行图像识别  
直接运行convention.ipynb文件即可

## 2. Yolo Part
**yolo**文件夹内运用Yolo进行的图像识别  
- `train.py` : 训练detect或classify模型  
- `classify_analyze.py` : 分析分类模型yolo产生的数据  
- `detect_analyze.py` : 分析探测模型yolo产生的数据  
- `check.ipynb` : 应用训练完的模型检测img中图片类别  

说明： 本实验对于分类模型统一使用[cifar10](https://github.com/ultralytics/assets/releases/download/v0.0.0/cifar10.zip)数据集，对于detect模型则使用[coco128](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip)进行训练。  
关于后续：不会做大幅改动了，可能会用不同的数据集再测试一下