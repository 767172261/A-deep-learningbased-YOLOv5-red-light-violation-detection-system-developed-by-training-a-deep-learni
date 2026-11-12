基于YOLOv5闯红灯及红绿灯检测系统
以下文字及代码仅供参考。
@[toc]
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/53fbe7ee22ef4ffd96e179a21e06c232.png)


Python基于YOLOv5的闯红灯检测系统
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a5560f22d9ad4102a4599648b4460c32.png)

1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b5024a0a25914df39edc077370ab85ed.png)
要构建一个基于YOLOv5的闯红灯检测系统，同学，我们需要完成以下几个步骤：

1. **数据准备**：收集和标注数据集。
2. **模型训练**：使用YOLOv5进行模型训练。
3. **UI设计**：使用PyQt5设计用户界面。
4. **实时检测**：集成模型到UI中，实现实时检测。

### 1. 数据准备

首先，同学需要一个包含车辆和交通信号灯的数据集。你可以从公开数据集中获取，或者自己采集并标注数据。


#### 这个你自己标注工具
- 使用LabelImg等工具进行数据标注。

### 2. 这个你自己模型训练

#### 安装YOLOv5
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

#### 准备配置文件
在`yolov5`目录下创建`data.yaml`：
```yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 2  # number of classes
names: ['car', 'traffic_light']
```

#### 训练模型
```bash
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --cfg cfg/training/yolov5s.yaml --weights yolov5s.pt --name yolov5s_results
```

### 3. UI设计

使用PyQt5设计用户界面。

#### 安装PyQt5
```bash
pip install PyQt5
```

#### 创建主窗口
```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
import numpy as np
from ultralytics import YOLO

class TrafficLightDetection(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = YOLO('best.pt')  # 加载训练好的模型

    def initUI(self):
        self.setWindowTitle("车辆红绿灯检测系统")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setText("请选择视频文件")
        self.label.setAlignment(Qt.AlignCenter)

        self.button_load = QPushButton("选择文件", self)
        self.button_load.clicked.connect(self.load_video)

        self.button_start = QPushButton("开始识别", self)
        self.button_start.clicked.connect(self.start_detection)

        self.button_exit = QPushButton("退出系统", self)
        self.button_exit.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button_load)
        layout.addWidget(self.button_start)
        layout.addWidget(self.button_exit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_video(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi)", options=options)
        if file_name:
            self.video_path = file_name
            self.cap = cv2.VideoCapture(file_name)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # 每秒更新30帧

    def start_detection(self):
        if not hasattr(self, 'video_path'):
            QMessageBox.warning(self, "警告", "请先选择视频文件！")
            return
        self.button_start.setEnabled(False)
        self.detect_traffic_lights()

    def detect_traffic_lights(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            results = self.model(frame)
            annotated_frame = results[0].plot()
            height, width, channel = annotated_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.label.setPixmap(QPixmap.fromImage(q_img))
            self.app.processEvents()  # 更新UI

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            self.button_start.setEnabled(True)
            return
        results = self.model(frame)
        annotated_frame = results[0].plot()
        height, width, channel = annotated_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(q_img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficLightDetection()
    window.show()
    sys.exit(app.exec_())
```

### 4. 实时检测

在上述代码中，我们实现了以下功能：

- 选择视频文件。
- 开始识别按钮触发检测过程。
- 实时显示检测结果。

### 总结

通过以上步骤，同学构建一个基于YOLOv5的闯红灯检测系统。这个系统包括数据准备、模型训练、UI设计和实时检测等功能。

仅供给我i你   参考。
