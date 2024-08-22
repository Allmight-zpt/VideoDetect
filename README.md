# 视频监控项目
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
- 截图视频片段，通过下面的命令可以截取视频中的片段用于测试，不需要每次都测试整个视频文件，提高效率。
- 参数说明：
  - input_file：输入文件路径
  - output_file：输出文件路径
  - scale_percent：输出文件相对于输入文件的缩放比例，50即为50%
  - start_frame：截取开始帧
  - end_frame：截取结束帧
```shell
python get_video.py \
       --input_file=xxx.mp4 \
       --output_file=xxx_division.mp4 \
       --scale_percent=50 \
       --start_frame=1500 \
       --end_frame=4000
```
- 读取视频保存每一帧图像作为训练数据，后续需要手动将所有数据进行分类，再使用分类后的数据进行分类模型的训练。
- 参数说明：
  - video_path：输入视频路径
  - output_folder：每一帧数据的输出路径
```shell
python get_data.py \
       --video_path=xxx.mp4 \
       --output_folder=output_folder
```
- 训练分类器
- 参数说明：
  - train_data：训练数据路径
  - batch_size：训练batch大小
  - num_classes：训练集类型数量
  - lr：学习率
  - num_epochs：训练epoch数量
```shell
python get_classifier.py \
       --train_data='./train_data' \
       --batch_size=32 \
       --num_classes=7 \
       --lr=0.001 \
       --num_epochs=10
       
```
- 开始检测
- 参数说明：
  - video_path：输入视频路径
  - scale_percent：输出文件相对于输入文件的缩放比例，50即为50%
  - start_frame：测试开始帧
  - num_classes：分类类型数量
```shell
python main.py
       --video_path=xxx.mp4 \
       --scale_percent=50 \
       --start_frame=1 \
       --num_classes=7
```