import cv2 as cv
from test_classifier import predict_image_cv2, load_model

# 1.获取视频对象
cap = cv.VideoCapture(r'.\正确完整操作_1500_4000_1920_1080_scale.mp4')
# 获取视频的总帧数
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
# 设置缩放比例
scale_percent = 50  # 例如，将视频帧缩小到原始尺寸的50%

# 设置开始帧
start_frame = 1
if start_frame > total_frames:
    print("start frame is large than total frames")
    exit()
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

# 读取模型
num_classes = 7
model = load_model(str(num_classes) + 'class.pth', num_classes)

# 文字颜色
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 255, 0)

# 判定结果
result = {
    'Mixing duration': {'status': 0, 'color': red},
    'Quartered': {'status': False, 'color': red},
    'Diagonal': {'status': False, 'color': red},
    'Quartered & Diagonal times': {'status': 0, 'color': red, 'flag': True},
    'Sieving': {'status': False, 'color': red},
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
                    if (predict_class == 0) & (result['Diagonal']['status']):
                        result['Quartered']['status'] = False
                        result['Quartered']['color'] = red
                        result['Diagonal']['status'] = False
                        result['Diagonal']['color'] = red
                        result['Mixing duration']['status'] = 0
                        result['Mixing duration']['color'] = red
                        result['Quartered & Diagonal times']['flag'] = True
                    # 四分
                    elif (predict_class == 2) & (result['Mixing duration']['color'] == green):
                        result['Quartered']['color'] = green
                        result['Quartered']['status'] = True

                    # 对角
                    elif (predict_class == 4) & result['Quartered']['status']:
                        result['Diagonal']['color'] = green
                        result['Diagonal']['status'] = True

                    # 过筛
                    elif predict_class == 6:
                        result['Sieving']['color'] = green
                        result['Sieving']['status'] = True

            else:
                counter[counter['last_frame']] = 0
                counter['last_frame'] = predict_class
        else:
            counter['last_frame'] = predict_class
        # 混土时长
        if result['Mixing duration']['color'] == red:
            result['Mixing duration']['status'] = round(counter[0] / 10, 2)
        if counter[0] == 150:
            result['Mixing duration']['color'] = green
        # 混杂次数
        if result['Quartered & Diagonal times']['flag'] & (result['Quartered']['color'] == green) & (result['Diagonal']['color'] == green):
            result['Quartered & Diagonal times']['status'] += 1
            result['Quartered & Diagonal times']['color'] = green
            result['Quartered & Diagonal times']['flag'] = False
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
        for idx, (k, v) in enumerate(result.items()):
            cv.putText(frame, k + ': ' + str(v['status']), (10, 70 + (idx + 1) * 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, v['color'], 1, cv.LINE_AA)

        cv.imshow('masked_frame', frame)
        cv.imshow("cropped_frame", cropped_frame)

    # 5.每一帧间隔为25ms
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# 6.释放视频对象
cap.release()
cv.destroyAllWindows()
