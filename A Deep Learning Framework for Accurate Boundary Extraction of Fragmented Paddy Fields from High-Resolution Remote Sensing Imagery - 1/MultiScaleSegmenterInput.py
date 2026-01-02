import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSegmenterInput(nn.Module):
    """
    Segmenter模型的多尺度输入处理模块
    将多尺度特征融合到Segmenter模型中
    """

    def __init__(self, embed_dim=768, patch_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 多尺度特征投影层
        self.scale_256_proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.scale_512_proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.scale_1024_proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 特征融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 14, 14))  # 对于512×512输入

    def forward(self, multi_scale_images):
        """
        处理多尺度输入

        Args:
            multi_scale_images: 字典，包含不同尺度的图像
                - 'scale_256': [B, 3, 256, 256]
                - 'scale_512': [B, 3, 512, 512]
                - 'scale_1024': [B, 3, 1024, 1024]

        Returns:
            x: 融合后的特征 [B, embed_dim, H//patch_size, W//patch_size]
        """
        batch_size = multi_scale_images['scale_512'].shape[0]

        # 1. 对每个尺度进行特征提取
        features = []

        # 尺度1: 256×256
        x_256 = self.scale_256_proj(multi_scale_images['scale_256'])  # [B, embed_dim, 16, 16]
        x_256 = F.interpolate(x_256, size=(32, 32), mode='bilinear', align_corners=False)
        features.append(x_256)

        # 尺度2: 512×512 (原始尺度)
        x_512 = self.scale_512_proj(multi_scale_images['scale_512'])  # [B, embed_dim, 32, 32]
        features.append(x_512)

        # 尺度3: 1024×1024
        x_1024 = self.scale_1024_proj(multi_scale_images['scale_1024'])  # [B, embed_dim, 64, 64]
        x_1024 = F.interpolate(x_1024, size=(32, 32), mode='bilinear', align_corners=False)
        features.append(x_1024)

        # 2. 融合多尺度特征
        fused_features = torch.cat(features, dim=1)  # [B, embed_dim*3, 32, 32]

        # 3. 通过融合卷积层
        x = self.fusion_conv(fused_features)  # [B, embed_dim, 32, 32]

        # 4. 添加位置编码
        pos_embed = self.pos_embed
        if x.shape[2:] != pos_embed.shape[2:]:
            pos_embed = F.interpolate(pos_embed, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + pos_embed

        # 5. 转换为序列格式 (为Transformer准备)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        return x, (H, W)  # 返回特征和空间尺寸


# 训练循环中的多尺度损失计算
class MultiScaleLoss(nn.Module):
    """
    多尺度损失函数
    在多个尺度上计算损失
    """

    def __init__(self, base_loss_fn, scale_weights=None):
        super().__init__()
        self.base_loss_fn = base_loss_fn

        # 设置各尺度的权重
        if scale_weights is None:
            self.scale_weights = {
                'scale_256': 0.2,
                'scale_512': 0.5,
                'scale_1024': 0.3
            }
        else:
            self.scale_weights = scale_weights

    def forward(self, preds_dict, targets_dict):
        """
        计算多尺度损失

        Args:
            preds_dict: 字典，包含不同尺度的预测
            targets_dict: 字典，包含不同尺度的目标
        """
        total_loss = 0

        for scale in ['scale_256', 'scale_512', 'scale_1024']:
            if scale in preds_dict and scale in targets_dict:
                # 计算当前尺度的损失
                scale_loss = self.base_loss_fn(preds_dict[scale], targets_dict[scale])

                # 加权求和
                total_loss += self.scale_weights[scale] * scale_loss

        return total_loss


# 数据预处理函数（用于R-EVI-NDWI特征）
def calculate_indices(image_tensor):
    """
    计算EVI和NDWI指数

    Args:
        image_tensor: 原始图像tensor [B, 4, H, W]，包含R, G, B, NIR波段

    Returns:
        indices_tensor: R-EVI-NDWI三通道tensor [B, 3, H, W]
    """
    B, C, H, W = image_tensor.shape

    # 假设输入通道顺序为: [R, G, B, NIR]
    R = image_tensor[:, 0:1, :, :]
    G = image_tensor[:, 1:2, :, :]
    B_band = image_tensor[:, 2:3, :, :]  # 避免与变量名冲突
    NIR = image_tensor[:, 3:4, :, :]

    # 计算EVI
    # EVI = 2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1)
    numerator = NIR - R
    denominator = NIR + 6 * R - 7.5 * B_band + 1
    EVI = 2.5 * numerator / (denominator + 1e-6)

    # 归一化EVI到[0, 1]
    EVI = (EVI - EVI.min()) / (EVI.max() - EVI.min() + 1e-6)

    # 计算NDWI
    # NDWI = (G - NIR) / (G + NIR)
    NDWI = (G - NIR) / (G + NIR + 1e-6)

    # 归一化NDWI到[0, 1]
    NDWI = (NDWI - NDWI.min()) / (NDWI.max() - NDWI.min() + 1e-6)

    # 组合成R-EVI-NDWI三通道
    indices_tensor = torch.cat([R, EVI, NDWI], dim=1)

    return indices_tensor


# 使用示例：在训练循环中
def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0

    for batch_idx, (images_dict, masks_dict, _) in enumerate(dataloader):
        # 将数据移动到设备
        for scale in images_dict:
            images_dict[scale] = images_dict[scale].to(device)
            masks_dict[scale] = masks_dict[scale].to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        preds_dict = model(images_dict)  # 模型返回多尺度预测

        # 计算损失
        loss = criterion(preds_dict, masks_dict)

        # 反向传播
        loss.backward()

        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新权重
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(dataloader)
    return avg_loss


if __name__ == "__main__":
    # 测试多尺度输入模块
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模拟数据
    batch_size = 4
    mock_images = {
        'scale_256': torch.randn(batch_size, 3, 256, 256),
        'scale_512': torch.randn(batch_size, 3, 512, 512),
        'scale_1024': torch.randn(batch_size, 3, 1024, 1024)
    }

    # 测试多尺度输入处理
    multi_scale_module = MultiScaleSegmenterInput(embed_dim=768, patch_size=16)
    multi_scale_module = multi_scale_module.to(device)

    for scale in mock_images:
        mock_images[scale] = mock_images[scale].to(device)

    features, spatial_size = multi_scale_module(mock_images)
    print(f"输出特征形状: {features.shape}")
    print(f"空间尺寸: {spatial_size}")

    # 测试指数计算
    mock_multispectral = torch.randn(batch_size, 4, 512, 512).to(device)
    indices = calculate_indices(mock_multispectral)
    print(f"R-EVI-NDWI形状: {indices.shape}")