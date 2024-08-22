import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default=r'D:\file\个人资料\科研\横向\制样视频20240715\不正确3.mp4')
parser.add_argument('--output_file', type=str, default='./不正确3_1920_1080.mp4')
parser.add_argument('--scale_percent', type=int, default=50)
parser.add_argument('--start_frame', type=int, default=1500)
parser.add_argument('--scale_percent', type=int, default=4000)

args = parser.parse_args()

# 视频文件路径
input_file = args.input_file
# 输出视频的文件名
output_file = args.output_file
# 设置缩放比例，将视频帧缩小到原始尺寸的50%
scale_percent = args.scale_percent
# 设置要截取的起始帧和结束帧
start_frame = args.start_frame
end_frame = args.end_frame

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

extract_video_section(input_file, output_file, start_frame, end_frame)