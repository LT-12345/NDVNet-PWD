import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba
import numpy as np

class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        return out

class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = nn.Linear(c_list_sum, c_list[4]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        att5 = self.sigmoid(self.att5(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                          nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5

        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


class GraphConv(MessagePassing):  # 图卷积层，基于 MessagePassing 框架实现的图神经网络层。
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__(aggr='mean')
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

        # 预定义的静态边
        self.static_edges = torch.tensor([
            # 水平方向连接
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             # 垂直方向连接
             0, 10, 20, 30, 40, 1, 11, 21, 31, 41],
            # 目标节点
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             10, 20, 30, 40, 50, 11, 21, 31, 41, 51]
        ])

        # 动态边的权重
        self.dynamic_weight = 0.3  # 可调节的权重参数

    def compute_hybrid_edges(self, x, k=4):
        """结合静态和动态边的方法"""
        # 获取静态边
        static_edges = self.static_edges.to(x.device)

        # 计算动态边
        dynamic_edges = compute_edges(x, k)

        # 合并边并去重
        combined_edges = torch.cat([static_edges, dynamic_edges], dim=1)
        combined_edges = torch.unique(combined_edges, dim=1)

        return combined_edges

    def forward(self, x, edge_index=None):
        identity = x

        # 如果没有提供edge_index，使用混合边
        if edge_index is None:
            edge_index = self.compute_hybrid_edges(x)

        row, col = edge_index
        deg = degree(col, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.linear(x)
        x = self.propagate(edge_index, x=x, norm=norm)
        x = F.gelu(x)
        x = self.norm(x)

        return x + identity

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


def compute_edges(x, k=4, use_spatial=False, spatial_weight=0.1):
    """
    计算动态边索引，可选择是否考虑空间结构
    
    Args:
        x: 输入特征张量 [batch_size, nodes, channels] 或 [nodes, channels]
        k: 每个节点连接的最近邻数量
        use_spatial: 是否使用空间距离信息
        spatial_weight: 空间距离的权重系数，越大表示空间位置越重要
        
    Returns:
        edge_index: 计算的边索引 [2, num_edges]
    """
    # 转换输入形状为 [nodes, channels]
    if len(x.shape) == 3:
        batch_size, nodes, channels = x.shape
        x = x.reshape(-1, channels)
    else:
        nodes = x.shape[0]
    
    # 计算特征相似度
    x_norm = F.normalize(x, p=2, dim=1)
    sim = torch.mm(x_norm, x_norm.t())
    
    # 添加空间信息（如果需要）
    if use_spatial:
        # 计算网格尺寸（假设是方形）
        size = int(np.sqrt(nodes))
        
        # 生成坐标
        pos_y = torch.div(torch.arange(nodes, device=x.device), size, rounding_mode='floor')
        pos_x = torch.arange(nodes, device=x.device) % size
        
        # 计算曼哈顿距离
        y_dist = (pos_y.unsqueeze(0) - pos_y.unsqueeze(1)).abs() 
        x_dist = (pos_x.unsqueeze(0) - pos_x.unsqueeze(1)).abs()
        spatial_dist = x_dist + y_dist
        
        # 结合特征相似度和空间距离
        sim = sim - spatial_weight * spatial_dist.float()
    
    # 选择最相似的k个邻居
    _, topk_indices = torch.topk(sim, k=min(k+1, nodes), dim=1)
    topk_indices = topk_indices[:, 1:] if topk_indices.size(1) > 1 else topk_indices
    
    # 构建边索引
    rows = torch.arange(nodes, device=x.device).view(-1, 1).expand(-1, topk_indices.size(1)).reshape(-1)
    cols = topk_indices.reshape(-1)
    
    # 构建双向边并去重
    edge_index = torch.stack([
        torch.cat([rows, cols]),
        torch.cat([cols, rows])
    ], dim=0)
    
    edge_index = torch.unique(edge_index, dim=1)
    
    return edge_index

class AdvancedAttention(nn.Module):
    """增强型多分支混合注意力机制，结合CBAM、ECA和多尺度特征"""
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.channels = channels
        reduced_channels = max(channels // reduction_ratio, 4)
        
        # 修改 ECA 实现
        t = int(abs(math.log(channels, 2) + 1))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # CBAM风格通道注意力
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        
        # 多尺度空间注意力
        self.spatial_conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.spatial_conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.spatial_fusion = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        
        # 通道-空间注意力交互
        self.gate = nn.Conv2d(channels, 2, kernel_size=1, bias=True)
        
        # 全局上下文编码
        self.context_modeling = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.LayerNorm([reduced_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1)
        )
        
        # 自适应学习参数
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, channels, height, width = x.shape
        
        # 修改 ECA 注意力计算
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        eca_att = self.sigmoid(y)
        
        # --- CBAM风格通道注意力 ---
        avg_out = self.channel_mlp(self.channel_avg_pool(x))
        max_out = self.channel_mlp(self.channel_max_pool(x))
        cbam_channel = self.sigmoid(avg_out + max_out)
        
        # 融合多种通道注意力
        channel_att = self.sigmoid(eca_att + cbam_channel)
        x_ca = x * channel_att
        
        # --- 多尺度空间注意力 ---
        # 聚合特征
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        
        # 多尺度卷积获取不同感受野
        spatial_out1 = self.spatial_conv1(spatial_in)
        spatial_out2 = self.spatial_conv2(spatial_in)
        spatial_out3 = self.spatial_conv3(spatial_in)
        
        # 融合多尺度特征
        spatial_out = self.spatial_fusion(
            torch.cat([spatial_out1, spatial_out2, spatial_out3], dim=1)
        )
        spatial_att = self.sigmoid(spatial_out)
        
        # --- 通道-空间交互门控 ---
        gate = self.sigmoid(self.gate(x_ca))
        gate_channels, gate_spatial = gate.chunk(2, dim=1)
        
        # 全局上下文编码
        context = self.context_modeling(
            self.channel_avg_pool(x_ca)
        )
        
        # --- 特征融合 ---
        # 应用通道和空间注意力
        refined = x_ca * spatial_att * gate_spatial + x_ca * gate_channels
        
        # 添加上下文信息和原始特征（带学习权重的残差连接）
        output = x + self.beta * refined + self.gamma * context
        
        return output




class UltraLight_VM_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, ndvi_evi_channels=2, c_list=[8,16,24,32,48,64],
                split_att='fc', bridge=True):
        super().__init__()
        
        self.bridge = bridge
        
        # NDVI/EVI特征提取 - 增强版
        self.ndvi_evi_conv = nn.Sequential(
            nn.Conv2d(ndvi_evi_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合层 - 增强版
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(input_channels + 16, c_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_list[0], c_list[0], kernel_size=1),
            nn.BatchNorm2d(c_list[0]),
            nn.ReLU(inplace=True)
        )
        
        # 添加通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_list[0], c_list[0] // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_list[0] // 4, c_list[0], kernel_size=1),
            nn.Sigmoid()
        )
        
        # 添加空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self.graph_conv = GraphConv(in_channels=c_list[4], out_channels=c_list[4])
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

        self.attn_module_out4 = AdvancedAttention(c_list[3])  # 32通道
        self.attn_module_out1 = AdvancedAttention(c_list[0])  # 8通道

        features_before = {}
        features_after = {}
        def register_hooks(model):
            def hook_before(m, i, o):
                features_before['attn'] = i[0].detach()
            def hook_after(m, i, o):
                features_after['attn'] = o.detach()
            
            model.attn_module_out1.register_forward_hook(hook_before)
            model.attn_module_out1.register_forward_hook(hook_after)

        register_hooks(self)

    def _create_edge_index(self, height, width):
        edge_index = []
        for i in range(height):
            for j in range(width):
                node = i * width + j
                if i > 0:
                    edge_index.append([node, (i - 1) * width + j])
                if i < height - 1:
                    edge_index.append([node, (i + 1) * width + j])
                if j > 0:
                    edge_index.append([node, i * width + (j - 1)])
                if j < width - 1:
                    edge_index.append([node, i * width + (j + 1)])
        return torch.tensor(edge_index, dtype=torch.long).t()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, ndvi_evi=None):
        # 如果提供了NDVI/EVI数据，进行多模态处理
        if ndvi_evi is not None:
            # 处理NDVI/EVI特征
            ndvi_evi_features = self.ndvi_evi_conv(ndvi_evi)
            
            # 合并特征
            x = torch.cat([x, ndvi_evi_features], dim=1)
            x = self.feature_fusion(x)
            
            # 应用通道注意力
            channel_weights = self.channel_attention(x)
            x = x * channel_weights
            
            # 应用空间注意力
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            spatial_weights = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
            x = x * spatial_weights
        
        # 继续原有的前向传播
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out

        if self.bridge: t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        

        # 应用GraphConv到t5

        batch_size, channels, height, width = t5.shape

        # 重塑张量为图处理格式

        graph_input = t5.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, H*W, C]

        # 处理每个batch

        processed = []

        for batch_idx in range(batch_size):

            # 应用图卷积

            graph_output = self.graph_conv(graph_input[batch_idx])  # [H*W, C]

            processed.append(graph_output)

        # 重新组合批次

        graph_output_batch = torch.stack(processed, dim=0)  # [B, H*W, C]

        # 转换回原始格式

        t5 = graph_output_batch.permute(0, 2, 1).view(batch_size, channels, height, width)  # [B, C, H, W]


        
        # 重塑张量
        batch_size, channels, height, width = out.shape
        out = out.view(batch_size, channels, -1).permute(0, 2, 1)  # 变为 (B, H*W, C)
        
        # 移除不必要的 stack 操作
        out = out.view(batch_size, height, width, -1).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 移除KAN处理，改为简单的激活函数处理
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32

        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode='bilinear',align_corners=True))
        out4 = self.attn_module_out4(out4)
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True))
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode='bilinear',align_corners=True))
        out1 = self.attn_module_out1(out1)
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W

        return torch.sigmoid(out0)

if __name__ == '__main__':
    uvm = UltraLight_VM_UNet()
    print(uvm)

