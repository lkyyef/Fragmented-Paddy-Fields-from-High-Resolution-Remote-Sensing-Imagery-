import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import os
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MultiScaleRiceDataset(Dataset):
    """
    多尺度水稻数据集
    输入图像尺寸：512×512
    多尺度输出：256×256, 512×512, 1024×1024
    """

    def __init__(self, data_dir, split='train', transform=None):
        """
        初始化数据集

        Args:
            data_dir: 数据目录
            split: 'train', 'val', 'test'
            transform: 数据增强变换
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # 加载数据路径
        self.image_paths = []
        self.mask_paths = []

        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            images_dir = os.path.join(split_dir, 'images')
            masks_dir = os.path.join(split_dir, 'masks')

            for img_name in os.listdir(images_dir):
                if img_name.endswith(('.png', '.jpg', '.tif')):
                    img_path = os.path.join(images_dir, img_name)
                    mask_name = img_name.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
                    mask_path = os.path.join(masks_dir, mask_name)

                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)

        print(f"Loaded {len(self.image_paths)} samples for {split} set")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像和掩码
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # 灰度图

        # 转换为numpy数组
        image = np.array(image)
        mask = np.array(mask)

        # 数据增强（仅训练时）
        if self.transform and self.split == 'train':
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # 转换为tensor
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).long()

        # 生成多尺度图像
        multi_scale_images = self.generate_multi_scale_images(image)
        multi_scale_masks = self.generate_multi_scale_masks(mask)

        return multi_scale_images, multi_scale_masks, self.image_paths[idx]

    def generate_multi_scale_images(self, image_tensor):
        """
        生成多尺度图像

        Args:
            image_tensor: 原始图像tensor [C, H, W]

        Returns:
            dict: 包含不同尺度图像的字典
        """
        scales = {
            'scale_256': (256, 256),
            'scale_512': (512, 512),  # 原始尺寸
            'scale_1024': (1024, 1024)
        }

        multi_scale = {}

        for scale_name, (target_h, target_w) in scales.items():
            if scale_name == 'scale_512':
                # 原始尺寸，不进行缩放
                scaled_img = image_tensor
            else:
                # 使用双线性插值进行缩放
                scaled_img = F.interpolate(
                    image_tensor.unsqueeze(0),  # 添加batch维度
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # 移除batch维度

            multi_scale[scale_name] = scaled_img

        return multi_scale

    def generate_multi_scale_masks(self, mask_tensor):
        """
        生成多尺度掩码

        Args:
            mask_tensor: 原始掩码tensor [H, W]

        Returns:
            dict: 包含不同尺度掩码的字典
        """
        scales = {
            'scale_256': (256, 256),
            'scale_512': (512, 512),  # 原始尺寸
            'scale_1024': (1024, 1024)
        }

        multi_scale = {}

        for scale_name, (target_h, target_w) in scales.items():
            if scale_name == 'scale_512':
                # 原始尺寸
                scaled_mask = mask_tensor
            else:
                # 使用最近邻插值进行缩放（保持类别标签不变）
                scaled_mask = F.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0).float(),  # 添加batch和channel维度
                    size=(target_h, target_w),
                    mode='nearest'
                ).squeeze().long()  # 移除维度并转换为long类型

            multi_scale[scale_name] = scaled_mask

        return multi_scale


def get_multi_scale_transforms():
    """
    获取数据增强变换

    Returns:
        train_transform: 训练集变换
        val_transform: 验证集变换
    """
    # 训练集数据增强
    train_transform = A.Compose([
        A.RandomResizedCrop(512, 512, scale=(0.8, 1.0)),  # 随机裁剪
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.VerticalFlip(p=0.5),  # 垂直翻转
        A.RandomRotate90(p=0.5),  # 随机旋转90度
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 验证集/测试集变换（只进行标准化）
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


class MultiScaleCollateFn:
    """
    自定义collate函数，处理多尺度数据
    """

    def __call__(self, batch):
        """
        处理一批数据

        Args:
            batch: 包含(images_dict, masks_dict, img_path)的列表

        Returns:
            batch_images: 多尺度图像字典
            batch_masks: 多尺度掩码字典
            img_paths: 图像路径列表
        """
        batch_images = {'scale_256': [], 'scale_512': [], 'scale_1024': []}
        batch_masks = {'scale_256': [], 'scale_512': [], 'scale_1024': []}
        img_paths = []

        for images_dict, masks_dict, img_path in batch:
            # 收集不同尺度的图像
            for scale in ['scale_256', 'scale_512', 'scale_1024']:
                batch_images[scale].append(images_dict[scale])
                batch_masks[scale].append(masks_dict[scale])

            img_paths.append(img_path)

        # 堆叠为batch
        for scale in batch_images.keys():
            batch_images[scale] = torch.stack(batch_images[scale], dim=0)
            batch_masks[scale] = torch.stack(batch_masks[scale], dim=0)

        return batch_images, batch_masks, img_paths


# 使用示例
if __name__ == "__main__":
    # 设置数据路径
    data_dir = "./rice_dataset"

    # 获取数据增强变换
    train_transform, val_transform = get_multi_scale_transforms()

    # 创建数据集
    train_dataset = MultiScaleRiceDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform
    )

    val_dataset = MultiScaleRiceDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform
    )

    # 创建数据加载器
    collate_fn = MultiScaleCollateFn()

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # 论文中batch_size=4
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 测试数据加载
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 获取一个batch的数据
    for batch_idx, (images_dict, masks_dict, img_paths) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  图像路径: {img_paths[:2]}")  # 打印前两个路径

        for scale in ['scale_256', 'scale_512', 'scale_1024']:
            print(f"  {scale} 图像尺寸: {images_dict[scale].shape}")
            print(f"  {scale} 掩码尺寸: {masks_dict[scale].shape}")
            print(f"  {scale} 像素值范围: [{images_dict[scale].min():.3f}, {images_dict[scale].max():.3f}]")

        if batch_idx == 0:  # 只查看第一个batch
            break