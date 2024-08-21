import torch
from PIL import Image
from simple_cnn import SimpleCNN, transform
import cv2

def load_model(model_path, num_classes):
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def process_image_cv2(image):
    # 将OpenCV图像（BGR）转换为PIL图像（RGB）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def predict_image(model_path, num_classes, image_path):
    model = load_model(model_path, num_classes)
    image = process_image(image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()


def predict_image_cv2(model, image):
    image = process_image_cv2(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

# 使用示例
if __name__ == '__main__':
    model_path = '4class.pth'
    image_path = './train_data/1/frame_0191.jpg'
    predicted_class_index = predict_image(model_path, 4, image_path)
    print(f'Predicted class index: {predicted_class_index}')
