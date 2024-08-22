
import cv2

# 视频文件路径
input_file = r'D:\file\个人资料\科研\横向\制样视频20240715\正确的制样视频.mp4'

# 输出视频的文件名
output_file = './正确完整操作_1500_4000_1920_1080_scale.mp4'
# 设置缩放比例
scale_percent = 50  # 例如，将视频帧缩小到原始尺寸的50%

def extract_video_section(input_file, output_file, start_frame, end_frame):
    # 打开视频文件
    cap = cv2.VideoCapture(input_file)

    # 获取视频的一些属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 缩放视频帧
    scale_width = int(width * scale_percent / 100)
    scale_height = int(height * scale_percent / 100)
    dim = (scale_width, scale_height)

    # 设置视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    out = cv2.VideoWriter(output_file, fourcc, fps, dim)

    # 设置起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 循环读取帧
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        # 调整帧的大小
        frame = cv2.resize(frame, dim)
        out.write(frame)  # 写入帧到输出文件

    # 释放资源
    cap.release()
    out.release()

extract_video_section(input_file, output_file, 1500, 4000)