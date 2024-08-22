import cv2 as cv
from test_classifier import predict_image_cv2, load_model
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default=r'.\正确完整操作_1500_4000_1920_1080_scale.mp4')
parser.add_argument('--scale_percent', type=int, default=50)
parser.add_argument('--start_frame', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=7)

args = parser.parse_args()
# 1.获取视频对象
cap = cv.VideoCapture(args.video_path)
# 获取视频的总帧数
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
# 设置缩放比例
scale_percent = args.scale_percent  # 例如，将视频帧缩小到原始尺寸的50%

# 设置开始帧
start_frame = args.start_frame
if start_frame > total_frames:
    print("start frame is large than total frames")
    exit()
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

# 读取模型
num_classes = args.num_classes
model = load_model(str(num_classes) + 'class.pth', num_classes)

# 文字颜色
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 255, 0)

# 判定结果
result = {
    '混合样本时间': {'status': 0, 'color': red},
    '是否均匀四分样本': {'status': False, 'color': red},
    '是否取对角样本': {'status': False, 'color': red},
    '成功四分采样的次数': {'status': 0, 'color': red, 'flag': True},
    '是否过筛': {'status': False, 'color': red},
}

# 检测结果
predict_dict = {
    0: "None",
    1: "Two",
    2: "Quartered",
    3: "Three",
    4: "Two & Diagonal",
    5: "Into Sieve",
    6: "Sieving",
}

# 计数器
counter = {
    'last_frame': -1,
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
}

# 2.判断是否读取成功
while (cap.isOpened()):
    # 3. 获取每一帧图像
    ret, frame = cap.read()
    # 4. 获取成功显示图像
    if ret:
        # 截取有效区域
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        # 设定截取区域为图片中间部分 (比如截取中心区域，宽度和高度分别为原图的一半)
        crop_width = width // 2
        crop_height = height // 2
        # 计算起始点 (top-left corner)
        start_x = (width - crop_width) // 2
        start_y = (height - crop_height)
        # 截取中间区域
        cropped_frame = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
        # 测试图片
        predict_class = predict_image_cv2(model, cropped_frame)
        '''
        更新数据
        '''
        if counter['last_frame'] != -1:
            if counter['last_frame'] == predict_class:
                counter[predict_class] += 1
                # 连续10帧才算有效
                if counter[predict_class] == 10:
                    # 重置
                    if (predict_class == 0) & (result['是否取对角样本']['status']):
                        result['是否均匀四分样本']['status'] = False
                        result['是否均匀四分样本']['color'] = red
                        result['是否取对角样本']['status'] = False
                        result['是否取对角样本']['color'] = red
                        result['混合样本时间']['status'] = 0
                        result['混合样本时间']['color'] = red
                        result['成功四分采样的次数']['flag'] = True
                    # 四分
                    elif (predict_class == 2) & (result['混合样本时间']['color'] == green):
                        result['是否均匀四分样本']['color'] = green
                        result['是否均匀四分样本']['status'] = True

                    # 对角
                    elif (predict_class == 4) & result['是否均匀四分样本']['status']:
                        result['是否取对角样本']['color'] = green
                        result['是否取对角样本']['status'] = True

                    # 过筛
                    elif (predict_class == 6) & (result['成功四分采样的次数']['status'] != 0):
                        result['是否过筛']['color'] = green
                        result['是否过筛']['status'] = True

            else:
                counter[counter['last_frame']] = 0
                counter['last_frame'] = predict_class
        else:
            counter['last_frame'] = predict_class
        # 混土时长
        if result['混合样本时间']['color'] == red:
            result['混合样本时间']['status'] = round(counter[0] / 10, 2)
        if counter[0] == 150:
            result['混合样本时间']['color'] = green
        # 混杂次数
        if result['成功四分采样的次数']['flag'] & (result['是否均匀四分样本']['color'] == green) & (result['是否取对角样本']['color'] == green):
            result['成功四分采样的次数']['status'] += 1
            result['成功四分采样的次数']['color'] = green
            result['成功四分采样的次数']['flag'] = False
        '''
        显示功能
        '''
        # 缩放视频帧
        scale_width = int(width * scale_percent / 100)
        scale_height = int(height * scale_percent / 100)
        dim = (scale_width, scale_height)
        frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        cropped_frame = cv.resize(cropped_frame, dim, interpolation=cv.INTER_AREA)

        # 获取当前帧数
        current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        # 计算进度百分比
        progress = current_frame / total_frames
        # 在图像上绘制进度条
        cv.rectangle(frame, (10, scale_height - 30), (int(10 + progress * (scale_width - 20)), scale_height - 10),
                     (0, 255, 0), -1)
        # 在图像上绘制文本
        frame_info = f'Frame: {current_frame}/{total_frames}'
        predict_info = f'Current type: {predict_dict[predict_class]}'
        cv.putText(frame, frame_info, (10, scale_height - 40), cv.FONT_HERSHEY_SIMPLEX, 1, blue, 2, cv.LINE_AA)
        cv.putText(frame, predict_info, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, blue, 1, cv.LINE_AA)
        # 假设 'SimHei.ttf' 是你本地的中文字体文件，路径需要替换为你实际的字体文件路径
        font_path = r'STXIHEI.TTF'
        # 尝试加载字体，调整字体大小
        try:
            font = ImageFont.truetype(font_path, 20)  # 25是字体大小，可以调整
        except IOError:
            print("字体文件加载失败，请检查字体路径！")
        for idx, (k, v) in enumerate(result.items()):
            # 将cv2的图片转为RGB格式
            cv2img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            pilimg = Image.fromarray(cv2img)
            # 创建绘图对象
            draw = ImageDraw.Draw(pilimg)
            # 绘制文本，确保文本为 Unicode 格式
            text = f"{k}: {str(v['status'])}"
            try:
                draw.text((10, 70 + (idx + 1) * 25), text, font=font, fill=v['color'])
            except UnicodeEncodeError:
                print(f"无法显示文本: {text}")
            # 将PIL图片转回OpenCV格式
            frame = cv.cvtColor(np.array(pilimg), cv.COLOR_RGB2BGR)
            # 修改前无法显示中文：
            # cv.putText(frame, k + ': ' + str(v['status']), (10, 70 + (idx + 1) * 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, v['color'], 1, cv.LINE_AA)

        cv.imshow('masked_frame', frame)
        cv.imshow("cropped_frame", cropped_frame)

    # 5.每一帧间隔为25ms
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# 6.释放视频对象
cap.release()
cv.destroyAllWindows()
