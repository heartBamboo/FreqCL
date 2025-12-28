import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


from backbone.ResNet50 import resnet50
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from utils import smart_joint
from utils.conf import base_path

from backbone.MambaVision import MambaVision
from backbone.DWTNet_MambaVision import DWTNet_MambaVision_B
from backbone.ResNet18 import resnet18
from backbone.DWTNet import DWT_resnet18


class IoT_NID_Train(Dataset):
    """
    Overrides dataset to change the getitem function.
    """
    IMG_SIZE = 32
    N_CLASSES = 16
    MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613)
    TEST_TRANSFORM = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __init__(self, root, train=True, transform=None,
                 target_transform=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # data_file = self.dataload(self.root)
        #
        # self.data = data_file['data']
        # self.targets = torch.from_numpy(data_file['targets']).long()
        # self.classes = data_file['classes']
        # self.segs = data_file['segs']
        # self._return_segmask = False

        data_np, label_list, class_list = self.load_data('IoT_NID', 128, train=self.train, train_ratio=0.8, seed=3407)

        self.data = data_np
        self.targets = label_list
        self.classes = class_list
        self._return_segmask = False

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.reshape(32, 32)

        # to return a PIL Image
        img = img.cpu().numpy()  # 将张量转换为 NumPy 数组
        img = Image.fromarray((img * 255).astype(np.uint8), mode='L')  # 灰度图像
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, not_aug_img, self.logits[index]] if hasattr(self, 'logits') else [
            img, target, not_aug_img]

        if self._return_segmask:
            raise "Unsupported segmentation output in training set!"

        return ret_tuple

    def __len__(self) -> int:
        return len(self.data)

    def load_data(self, dataset_name: str, packet_length: int, train: bool = True, train_ratio: float = 0.8,
                  seed: int = 3407):
        dataset_dir = f'/data/users/lph/projects/IIOT_Incremental_Learning/data/{dataset_name}'

        # 使用 torch.load 加载 pt 文件
        data = torch.load(f'{dataset_dir}/data.pt')
        label = torch.load(f'{dataset_dir}/label.pt')
        classes = torch.load(f'{dataset_dir}/label_map.pt').values()

        # 如果需要，reshape data
        data = data.view(len(data), 32, 32)

        # label 转为 numpy 数组方便后续处理（如果你后续代码需要）
        label = label.numpy()

        # 固定随机种子，保证划分一致
        np.random.seed(seed)
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        train_size = int(len(data) * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        if train:
            selected_indices = train_indices
        else:
            selected_indices = test_indices

        data = data[selected_indices]
        label = label[selected_indices]

        unique, counts = np.unique(label, return_counts=True)
        print(f"{'训练集' if train else '测试集'}类别分布统计:")
        for cls, cnt in zip(unique, counts):
            print(f"类别 {cls}: {cnt} 个样本")

        return data, label, classes

    def get_class_distribution(self):
        """
        统计当前数据集中每个类别的样本数量。

        Returns:
            dict: {class_label: count}
        """
        # self.targets 是 tensor，先转成 numpy
        labels_np = self.targets.numpy()
        unique, counts = np.unique(labels_np, return_counts=True)
        distribution = dict(zip(unique, counts))
        return distribution


class IoT_NID_Test(IoT_NID_Train):
    """Base IoT_NID dataset."""

    def __init__(self, root, train=False, transform=None, target_transform=None) -> None:
        super().__init__(root, train=False, transform=transform,
                         target_transform=target_transform)

    def __getitem__(self, index: int, ret_segmask=False) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.reshape(32, 32)

        # to return a PIL Image
        img = img.cpu().numpy()  # 将张量转换为 NumPy 数组
        img = Image.fromarray((img * 255).astype(np.uint8), mode='L')  # 灰度图像

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, self.logits[index]] if hasattr(self, 'logits') else [img, target]

        if ret_segmask or self._return_segmask:
            seg = self.segs[index]
            seg = Image.fromarray(seg, mode='L')
            seg = transforms.ToTensor()(transforms.CenterCrop((IoT_NID_Train.IMG_SIZE, IoT_NID_Train.IMG_SIZE))(seg))[0]
            ret_tuple.append((seg > 0).int())

        return ret_tuple


class Sequential_IoT_NID(ContinualDataset):
    """Sequential IOT_NID Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data.
        TEST_TRANSFORM (torchvision.transforms): transformation to apply to the test data.
    """
    NAME = 'IoT_NID'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    SIZE = (IoT_NID_Train.IMG_SIZE, IoT_NID_Train.IMG_SIZE)
    # MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613) # RGB
    MEAN, STD = (0.4856,), (0.2272,)  # 单通道的均值和标准差
    TRANSFORM = transforms.Compose([
        #transforms.RandomCrop(IoT_NID_Train.IMG_SIZE, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
        ])
    TEST_TRANSFORM = IoT_NID_Train.TEST_TRANSFORM

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        IoT_NID_Train_dataset = IoT_NID_Train(base_path() + 'IoT_NID', train=True, transform=transform)
        IoT_NID_Test_dataset = IoT_NID_Test(base_path() + 'IoT_NID', train=False, transform=test_transform)

        train, test = store_masked_loaders(
            IoT_NID_Train_dataset, IoT_NID_Test_dataset, self)

        train_dist = IoT_NID_Train_dataset.get_class_distribution()
        test_dist = IoT_NID_Test_dataset.get_class_distribution()
        print("训练类别分布统计：")
        for cls, count in train_dist.items():
            print(f"类别 {cls}: {count} 个样本")

        print("测试类别分布统计：")
        for cls, count in test_dist.items():
            print(f"类别 {cls}: {count} 个样本")

        return train, test

    # def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    #     transform = self.TRANSFORM
    #     test_transform = transforms.Compose(
    #         [transforms.ToTensor(), self.get_normalization_transform()])
    #
    #     # 加载完整数据集
    #     full_train_dataset = IoT_NID_Train(base_path() + 'IoT_NID', train=True, transform=transform)
    #     full_test_dataset = IoT_NID_Test(base_path() + 'IoT_NID', train=False, transform=test_transform)
    #
    #     # 当前任务类别范围
    #     task_id = self._current_task if hasattr(self, '_current_task') else 0  # 你需要传入或设置当前任务id
    #     task_classes = list(range(task_id * self.N_CLASSES_PER_TASK, (task_id + 1) * self.N_CLASSES_PER_TASK))
    #
    #     # 过滤并映射标签
    #     filtered_train_dataset = self.filter_dataset(full_train_dataset, task_classes, transform)
    #     filtered_test_dataset = self.filter_dataset(full_test_dataset, task_classes, test_transform)
    #
    #     # 传入过滤后的数据集，调用原来的 store_masked_loaders
    #     train_loader, test_loader = store_masked_loaders(filtered_train_dataset, filtered_test_dataset, self)
    #
    #     # 打印类别分布
    #     train_dist = filtered_train_dataset.get_class_distribution()
    #     test_dist = filtered_test_dataset.get_class_distribution()
    #     print("训练类别分布统计：")
    #     for cls, count in train_dist.items():
    #         print(f"类别 {cls}: {count} 个样本")
    #
    #     print("测试类别分布统计：")
    #     for cls, count in test_dist.items():
    #         print(f"类别 {cls}: {count} 个样本")
    #
    #     return train_loader, test_loader
    #
    # def filter_dataset(self, dataset, task_classes, transform):
    #     data = dataset.data
    #     labels = dataset.targets
    #
    #     if hasattr(labels, 'numpy'):
    #         labels_np = labels.numpy()
    #     else:
    #         labels_np = labels
    #
    #     print("原始标签类别:", np.unique(labels_np))
    #     print("任务类别:", task_classes)
    #
    #     mask = np.isin(labels_np, task_classes)
    #     print(f"过滤后样本数: {mask.sum()}")
    #
    #     filtered_data = data[mask]
    #     filtered_labels = labels_np[mask]
    #
    #     class_to_idx = {cls: idx for idx, cls in enumerate(task_classes)}
    #     remapped_labels = np.array([class_to_idx[l] for l in filtered_labels])
    #
    #     class FilteredDataset(torch.utils.data.Dataset):
    #         def __init__(self, data, targets, transform):
    #             self.data = data
    #             self.targets = targets
    #             self.transform = transform
    #
    #         def __len__(self):
    #             return len(self.targets)
    #
    #         def __getitem__(self, idx):
    #             x = self.data[idx]
    #             y = self.targets[idx]
    #
    #             # 如果 x 是 numpy ndarray，转成 PIL Image
    #             if isinstance(x, np.ndarray):
    #                 x = Image.fromarray(x)
    #
    #             if self.transform:
    #                 x = self.transform(x)
    #             return x, y
    #
    #         def get_class_distribution(self):
    #             unique, counts = np.unique(self.targets, return_counts=True)
    #             return dict(zip(unique, counts))
    #
    #     return FilteredDataset(filtered_data, remapped_labels, transform)


    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), Sequential_IoT_NID.TRANSFORM])
        return transform

    @staticmethod
    def get_DWT_transform():
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(Sequential_IoT_NID.MEAN, Sequential_IoT_NID.STD)])
        dwt_transform = transforms.Compose(
            [transforms.ToPILImage(), test_transform])
        return dwt_transform

    @staticmethod
    def get_backbone():
        #return resnet18(Sequential_IoT_NID.N_CLASSES_PER_TASK * Sequential_IoT_NID.N_TASKS)
        num_classes = Sequential_IoT_NID.N_CLASSES_PER_TASK * Sequential_IoT_NID.N_TASKS
        model_path = "/tmp/mamba_vision_B.pth.tar"
        depths = [3, 3, 10, 5]
        num_heads = [2, 4, 8, 16]
        window_size = [4, 4, 2, 2]
        dim = 128
        in_dim = 64
        mlp_ratio = 4
        resolution = 224
        drop_path_rate = 0.1
        layer_scale = 1e-5
        layer_scale_conv = None
        num_classes = 10

        in_chans = 1
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.

        model = MambaVision(
            dim,
            in_dim,
            depths,
            window_size,
            mlp_ratio,
            num_heads,
            drop_path_rate,
            in_chans,
            num_classes,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            layer_scale,
            layer_scale_conv,
        )
        return model

    @staticmethod
    def get_DWT_backbone():
        #return DWT_resnet18(Sequential_IoT_NID.N_CLASSES_PER_TASK * Sequential_IoT_NID.N_TASKS)
        num_classes = Sequential_IoT_NID.N_CLASSES_PER_TASK * Sequential_IoT_NID.N_TASKS
        #MambaVision_B
        model = DWTNet_MambaVision_B(
            wavename='haar',
            pretrained=False,
        )
        return model

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(
            Sequential_IoT_NID.MEAN, Sequential_IoT_NID.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(Sequential_IoT_NID.MEAN, Sequential_IoT_NID.STD)
        return transform

    @staticmethod
    def get_batch_size():
        return 64

    @staticmethod
    def get_minibatch_size():
        return 64

    @staticmethod
    def get_epochs():
        return 30
