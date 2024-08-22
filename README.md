# 视频监控项目
___
监控视频中的实验人员操作是否合理。
将视频中的操作分割为不同阶段，每个阶段的动作构建为一类数据，根据构建的数据集训练分类器。
读取视频，使用分类器对每一帧进行分类划分为不同阶段。
根据分类结果判断操作是否合理。
___
### 正确操作检测标准
- 混土时长 > 15s
- 是否将实验材料四等分
- 是否取对焦线上的实验材料进行混合
- 统计混杂次数
- 是否过筛

### 效果
![img.png](assets%2Fimg.png)

### 界面说明
- Mixing duration：混土时长
- Quartered：是否四等分
- Diagonal：是否取对角线
- Quartered & Diagonal times：混杂次数
- Sieving：是否过筛
- 绿色即为达标，红色为不达标

### 快速使用
- 截图视频片段
```shell
python get_video.py
```
- 读取视频保存每一帧图像作为训练数据
```shell
python get_data.py
```
- 训练分类器
```shell
python get_classifier.py
```
- 开始检测
```shell
python main.py
```