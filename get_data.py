import cv2
import os

# 视频文件路径
video_path = r'.\正确的制样视频_scale.mp4'

# 保存帧的文件夹路径
output_folder = 'raw_data'
os.makedirs(output_folder, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

frame_count = 0

# 循环读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 获取帧的尺寸
    height, width, _ = frame.shape

    # 设定截取区域为图片中间部分 (比如截取中心区域，宽度和高度分别为原图的一半)
    crop_width = width // 2
    crop_height = height // 2

    # 计算起始点 (top-left corner)
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height)

    # 截取中间区域
    cropped_frame = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # 保存截取后的图像
    frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, cropped_frame)

    frame_count += 1

# 释放视频对象
cap.release()

print(f'Total frames extracted: {frame_count}')
