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




class ISCXTor2016(Dataset):
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

        data_np, label_list, class_list = self.load_data('ISCX_Tor_2016', 128)

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


    def load_data(self, dataset_name: str, packet_length: int, seed: int = 3407, scene_index: int = 0):
        if dataset_name == 'ISCX-Bot-2014':
            dataset_dir = f'./data/{dataset_name}/old_processed'
        else:
            dataset_dir = f'./data/{dataset_name}/processed'
        if dataset_name == 'CTU-13':
            botnet_data = torch.load(
                f'./data/{dataset_name}/processed/scene_{scene_index}_p{packet_length - 32}_w8.pt')[:, :,
                          :packet_length]
            benign_data = np.load(f'./data/ISCX-Bot-2014/processed/stream_feat_p{packet_length - 32}_w8.npy')[:, :,
                          :packet_length]
            shuffle_index = np.random.permutation(len(benign_data))[:len(botnet_data)]
            benign_data = benign_data[shuffle_index]
            data = np.concatenate((botnet_data, benign_data))

            label = np.concatenate((
                np.ones(len(botnet_data), dtype=np.int8),
                np.zeros(len(benign_data), dtype=np.int8)
            ))
        else:
            # data = np.load(f'{dataset_dir}/stream_feat_p{packet_length - 32}_w8.npy')[:, :, :packet_length]
            data = np.load(f'{dataset_dir}/stream_feat.npy')[:, :, :packet_length]
            label = np.load(f'{dataset_dir}/label.npy')
            classes = torch.load(f'{dataset_dir}/label_map.pt').values()

            # data = data[::-1].copy()
            # label = label[::-1].copy()
            # classes = list(classes)[::-1]

            data = torch.as_tensor(data, dtype=torch.float32).view(len(data), 32, 32)

            num_classes = len(np.unique(label))

            return data, label, classes




class MyTor2016(ISCXTor2016):
    """Base Tor2016 dataset."""

    def __init__(self, root, train=True, transform=None, target_transform=None) -> None:
        super().__init__(root, train=train, transform=transform,
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
            seg = transforms.ToTensor()(transforms.CenterCrop((ISCXTor2016.IMG_SIZE, ISCXTor2016.IMG_SIZE))(seg))[0]
            ret_tuple.append((seg > 0).int())

        return ret_tuple


class SequentialTor2016(ContinualDataset):
    """Sequential CUB200 Dataset.

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
    NAME = 'ISCX_Tor_2016'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    SIZE = (ISCXTor2016.IMG_SIZE, ISCXTor2016.IMG_SIZE)
    # MEAN, STD = (0.4856, 0.4994, 0.4324), (0.2272, 0.2226, 0.2613) # RGB
    MEAN, STD = (0.4856,), (0.2272,)  # 单通道的均值和标准差
    TRANSFORM = transforms.Compose([
        #transforms.RandomCrop(ISCXTor2016.IMG_SIZE, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
        ])
    TEST_TRANSFORM = ISCXTor2016.TEST_TRANSFORM

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = ISCXTor2016(base_path() + 'ISCX_Tor_2016', train=True, transform=transform)
        test_dataset = MyTor2016(base_path() + 'ISCX_Tor_2016', train=False, transform=test_transform)

        train, test = store_masked_loaders(
            train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialTor2016.TRANSFORM])
        return transform

    @staticmethod
    def get_DWT_transform():
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(SequentialTor2016.MEAN, SequentialTor2016.STD)])
        dwt_transform = transforms.Compose(
            [transforms.ToPILImage(), test_transform])
        return dwt_transform

    @staticmethod
    def get_backbone():
        #return resnet18(SequentialTor2016.N_CLASSES_PER_TASK * SequentialTor2016.N_TASKS)
        num_classes = SequentialTor2016.N_CLASSES_PER_TASK * SequentialTor2016.N_TASKS
        model_path = "/tmp/mamba_vision_B.pth.tar"
        depths = [3, 3, 10, 5]
        num_heads = [2, 4, 8, 16]
        window_size = [4, 4, 2, 2]
        dim = 128
        in_dim = 64
        mlp_ratio = 4
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
        #return DWT_resnet18(SequentialTor2016.N_CLASSES_PER_TASK * SequentialTor2016.N_TASKS)
        num_classes = SequentialTor2016.N_CLASSES_PER_TASK * SequentialTor2016.N_TASKS
        model = DWTNet_MambaVision_B(
            wavename='haar',
            pretrained=True,
        )
        return model

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(
            SequentialTor2016.MEAN, SequentialTor2016.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialTor2016.MEAN, SequentialTor2016.STD)
        return transform

    @staticmethod
    def get_batch_size():
        return 16

    @staticmethod
    def get_epochs():
        return 30
