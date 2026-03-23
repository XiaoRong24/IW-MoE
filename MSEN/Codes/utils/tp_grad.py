
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd  # 👈 导入 autograd
import numpy as np
import copy
import random
from itertools import combinations
import matplotlib.pyplot as plt # 新增

class TaskPCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        self.conflict_stats = {
            'layer_conflicts': [],  # 各层冲突次数
            'global_conflict': [],  # 全局冲突比例
            'grad_norms': []  # 梯度范数变化
        }
        self.writer = None  # 后续绑定TensorBoard writer

    def set_writer(self, writer):
        self.writer = writer

    def _record_stats(self, grads):
        """记录梯度冲突统计信息"""
        batch_conflicts = []
        for g1, g2 in combinations(grads, 2):
            cos_sim = F.cosine_similarity(g1.flatten(), g2.flatten(), dim=0)
            batch_conflicts.append((cos_sim < 0.5).float())

        conflict_matrix = torch.stack(batch_conflicts).mean(dim=0)
        self.conflict_stats['layer_conflicts'].append(conflict_matrix)
        self.conflict_stats['global_conflict'].append(conflict_matrix.mean())

        # 记录梯度范数
        self.conflict_stats['grad_norms'].append(
            [g.norm().item() for g in grads]
        )

    def log_to_tensorboard(self, global_step):
        if self.writer is None:
            return

        # 记录全局冲突率
        self.writer.add_scalar(
            'PCGrad/global_conflict_rate',
            self.conflict_stats['global_conflict'][-1],
            global_step
        )

        # 记录各层冲突热力图 (使用 matplotlib 和 add_figure 替代 add_heatmap)
        if len(self.conflict_stats['layer_conflicts']) > 0:
            # 1. 获取最新的冲突张量（一维向量）
            conflict_vector = self.conflict_stats['layer_conflicts'][-1].cpu().numpy()

            # 2. **FIX: 检查 NumPy 数组的尺寸**
            # 如果数组是标量 (shape=()) 或长度为 0，则跳过绘图
            if conflict_vector.ndim == 0 or conflict_vector.size == 0:
                # 可以选择记录一个提示或直接退出此段
                print(f"Warning: Conflict vector is empty or scalar at step {global_step}. Skipping heatmap log.")
                return  # 提前退出，避免绘图

            # 3. 创建 Matplotlib Figure
            fig, ax = plt.subplots(figsize=(8, 1))  # 创建一个扁平的图表

            # 确保冲突向量是非空的 (这一行不再需要，因为上面已经检查了)
            # if conflict_vector.size > 0:

            # 使用 imshow 绘制热图，形状必须是二维的 (1, N)
            im = ax.imshow(conflict_vector.reshape(1, -1), cmap='coolwarm', aspect='auto',
                           vmin=0, vmax=1)  # 假设冲突率在 0 到 1 之间

            # 配置轴标签 (len() 现在是安全的，因为我们已经检查了尺寸)
            ax.set_xticks(np.arange(len(conflict_vector)))
            ax.set_xticklabels([f'Layer_{i}' for i in range(len(conflict_vector))], rotation=45, ha="right")
            ax.set_yticks([])  # 移除 Y 轴
            ax.set_title("Per-Layer Gradient Conflict Rate")

            # 添加颜色条
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
            cbar.set_label('Conflict Rate ($\cos(\mathbf{g}_i, \mathbf{g}_j) < 0.5$)', rotation=270, labelpad=15)

            # 4. 使用 add_figure 记录图表
            self.writer.add_figure(
                'PCGrad/layer_conflicts',
                fig,
                global_step
            )
            plt.close(fig)  # 释放内存

        # 记录梯度范数
        for i, norm in enumerate(self.conflict_stats['grad_norms'][-1]):
            self.writer.add_scalar(
                f'PCGrad/grad_norm/task_{i}',
                norm,
                global_step
            )
    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def _calculate_expert_weights(self, router_logits, num_tasks):
        """计算任务-专家权重"""
        if router_logits is None:
            return None

        router_probs = F.softmax(router_logits, dim=-1)
        avg_probs = torch.mean(router_probs, dim=0)

        # 添加维度检查
        if avg_probs.dim() > 1:
            print(f"Warning: router_probs has unexpected shape {router_probs.shape}")
            avg_probs = avg_probs.mean(dim=0)  # 额外平均处理

        # 使用传入的 num_tasks 参数而不是从优化器获取
        if len(avg_probs) != num_tasks:
            # print(f"Adjusting weights from {len(avg_probs)} to {num_tasks}")
            # 创建均匀分布的任务权重
            avg_probs = torch.ones(num_tasks, device=router_logits.device) / num_tasks

        return avg_probs

    def pc_backward(self, objectives, router_logits=None):
        """
        使用 torch.autograd.grad 避免显存泄漏，并执行梯度投影。
        """
        # 1. 包装并计算梯度 (使用 autograd.grad 避免 retain_graph=True)
        grads, shapes, has_grads = self._pack_grad(objectives)

        # 2. 计算任务-专家权重 (传入任务数量)
        task_weights = self._calculate_expert_weights(router_logits, len(objectives))

        # 3. 执行梯度投影
        pc_grad = self._project_conflicting(grads, has_grads, task_weights)  # pc_grad 是扁平张量

        # 4. 设置梯度（在设置时进行解扁平化）
        # 传入扁平张量和形状信息
        self._set_grad(pc_grad, shapes[0])
        return

    def _project_conflicting(self, grads, has_grads, task_weights):
        """
        执行梯度投影，支持任务权重加权

        Args:
            grads: 各任务的扁平化梯度列表 [task1_grad, task2_grad, ...]
            has_grads: 各参数的梯度存在标记
            task_weights: 从router_logits计算的任务权重 (shape: [num_tasks])
        """
        # 1. 准备投影用梯度副本
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad = [g.clone() for g in grads]
        num_tasks = len(grads)

        # 2. 如果存在任务权重，进行归一化处理
        if task_weights is not None:
            # 确保权重与任务数匹配
            assert len(task_weights) == num_tasks, \
                f"Task weights dim {len(task_weights)} != num tasks {num_tasks}"

            # 归一化为总和为1的权重
            task_weights = F.softmax(task_weights, dim=0)
            # print(f"Applying task weights: {task_weights.tolist()}")  # 调试用

        # 3. 带权重的梯度投影
        for idx in range(num_tasks):
            g_i = pc_grad[idx]

            # 创建其他任务的索引列表（排除当前任务）
            other_indices = [j for j in range(num_tasks) if j != idx]
            random.shuffle(other_indices)

            for j in other_indices:
                g_j = grads[j]  # 使用原始梯度进行比较

                # 计算原始点积
                g_i_g_j = torch.dot(g_i, g_j)

                # 如果存在任务权重，调整投影强度
                if task_weights is not None:
                    # 获取当前任务对的权重乘积
                    weight_factor = task_weights[idx] * task_weights[j]
                    g_i_g_j *= weight_factor * 2  # 乘2保持强度平衡

                if g_i_g_j < 0:  # 仅当仍有冲突时才投影
                    projection = g_i_g_j * g_j / (g_j.norm() ** 2 + 1e-8)
                    pc_grad[idx] -= projection

        # 4. 加权合并梯度
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

        if task_weights is None:
            # 无权重时的常规合并
            if self._reduction == 'mean':
                merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
            else:
                merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            # 带权重的合并 (加权平均)
            weighted_grads = torch.stack([g[shared] * task_weights[i]
                                          for i, g in enumerate(pc_grad)])
            merged_grad[shared] = weighted_grads.sum(dim=0)

        # 处理非共享参数
        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)

        return merged_grad
    def _set_grad(self, grads, shapes):
        """
        设置梯度。grads是扁平的合并梯度张量，shapes是参数的形状列表。
        """
        idx = 0
        shape_idx = 0  # 用于跟踪 shapes 列表
        for group in self._optim.param_groups:
            for p in group['params']:
                # 仅对需要梯度的参数进行操作
                if p.requires_grad:
                    shape = shapes[shape_idx]
                    length = np.prod(shape)

                    # 关键修复点：从扁平张量中切片并使用 .view() 重塑
                    p.grad = grads[idx:idx + length].view(shape).to(p.device)

                    idx += length
                    shape_idx += 1
        return

    def _pack_grad(self, objectives):
        """
        使用 torch.autograd.grad 计算并扁平化梯度，解决显存泄漏问题。
        """
        grads, shapes, has_grads = [], [], []

        # 收集所有需要梯度的参数
        params = [p for group in self._optim.param_groups for p in group['params'] if p.requires_grad]

        # 在这里添加打印，确认 params 列表非空
        # print(f"Number of trainable parameters: {len(params)}")
        # print(f"Total elements: {sum(p.numel() for p in params)}")

        # 收集所有参数的形状
        all_shapes = [p.shape for p in params]

        for obj in objectives:
            # 使用 torch.autograd.grad 来计算独立梯度
            task_grads = torch.autograd.grad(
                obj,
                params,
                allow_unused=True,
                retain_graph=True
            )

            packed_grad = []
            packed_has_grad = []

            for p_grad, p in zip(task_grads, params):
                if p_grad is None:
                    # 如果参数没有梯度，使用零张量和零标记
                    packed_grad.append(torch.zeros_like(p).to(p.device))
                    packed_has_grad.append(torch.zeros_like(p).to(p.device))
                else:
                    packed_grad.append(p_grad.clone())
                    packed_has_grad.append(torch.ones_like(p).to(p.device))

            flat_grad = self._flatten_grad(packed_grad)
            # 在这里添加打印，确认 flat_grad 的大小
            # print(f"Flattened gradient size: {flat_grad.numel()}")

            # 扁平化并存储
            grads.append(self._flatten_grad(packed_grad))
            has_grads.append(self._flatten_grad(packed_has_grad))

        # 只需要存储一组形状信息
        shapes.append(all_shapes)

        return grads, shapes, has_grads

    # 保持 _flatten_grad 不变
    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad