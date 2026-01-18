from ultralytics import YOLO

# 1) 选模型：可用预训练权重(.pt)或结构(.yaml)
model = YOLO('/home/user/cyshi_lx/ultralytics/ultralytics/cfg/models/v8/BD-YOLO.yaml')  # 或 'yolov8s.yaml'

# 2) 训练
model.train(
    data='/home/user/cyshi_lx/ultralytics/ultralytics/cfg/datasets/SASI.yaml',  # 数据集YAML
    imgsz=840, epochs=300, batch=16, device='0',
    optimizer='SGD', lr0=0.005, momentum=0.937, weight_decay=5e-4,
    project='/home/user/cyshi_lx/ultralytics/ultralytics/work_dir/yuan', name='bd-yolo', verbose=True
)


