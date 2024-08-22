import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm  # 导入 tqdm
from simple_cnn import SimpleCNN, transform
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default=r'./train_data')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=10)

args = parser.parse_args()

# 加载数据集
train_dataset = datasets.ImageFolder(root=args.train_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# 实例化模型
num_classes = args.num_classes
model = SimpleCNN(num_classes=num_classes)

# 使用GPU进行训练（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 训练模型
num_epochs = args.num_epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # 使用 tqdm 包装 DataLoader 以显示进度条
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

print('Training complete.')

# 保存模型
torch.save(model.state_dict(), str(num_classes) + 'class.pth')
print('Model saved as simple_cnn.pth')
