import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 设置数据集路径和归一化参数
CIFAR_PATH = "../data/CIFAR-10"
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

# 定义数据预处理步骤
transform_train = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(15), # 数据增强
transforms.ToTensor(),
transforms.Normalize(mean, std)
])
transform_test = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean, std)
])

# 加载 CIFAR-10 训练集和测试集
cifar100_training = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=True, download=True, transform=transform_train)
cifar100_testing = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=False, download=True, transform=transform_test)

# 创建数据加载器
trainloader = DataLoader(cifar100_training, batch_size=64, shuffle=True)
testloader = DataLoader(cifar100_testing, batch_size=100, shuffle=False)