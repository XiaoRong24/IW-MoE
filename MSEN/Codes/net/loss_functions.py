import torch
import numpy as np
import torch.nn as nn
from math import exp
import torch.nn.functional as F
from torchvision.models import vgg16, vgg19
from torch.autograd import Variable
from skimage.metrics import structural_similarity as compare_ssim
import utils.constant as constant

grid_w = constant.GRID_W
grid_h = constant.GRID_H
gpu_device = constant.GPU_DEVICE

min_w = (512 / grid_w) / 8
min_h = (384 / grid_h) / 8


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class TV_Loss(torch.nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IF):
        r = IA - IF
        h = r.shape[2]
        w = r.shape[3]
        tv1 = torch.pow((r[:, :, 1:, :] - r[:, :, :h - 1, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:] - r[:, :, :, :w - 1]), 2).mean()

        return tv1 + tv2


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            window = window.to(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,img1,img2):
        device = img1.device
        b, c, h, w = img1.shape
        kernel = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) \
            .to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        f1_grad =  F.conv2d(img1, kernel,padding=1,groups=c)
        f2_grad =  F.conv2d(img2, kernel,padding=1,groups=c)
        totalGradLoss = intensity_loss(gen_frames=f1_grad, gt_frames=f2_grad, l_num=2)
        return totalGradLoss

class VGG(nn.Module):
    def __init__(self, layer_indexs):
        super(VGG, self).__init__()
        layers = []
        in_dim = 3
        out_dim = 64
        self.layer_indexs = layer_indexs
        for i in range(16):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dim = out_dim
            if i == 1 or i == 3 or i == 7 or i == 11 or i == 15:
                layers += [nn.MaxPool2d(2, 2)]
                if i != 11:
                    out_dim *= 2
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        out = []
        for i in range(len(self.layer_indexs)):
            if i == 0:
                x = self.features[:self.layer_indexs[0]+1](x)
            else:
                x = self.features[self.layer_indexs[i-1]+1:self.layer_indexs[i]+1](x)
            out.append(x)
        # print("out:",len(out))
        return out


class PerceptualLoss(nn.Module):
    def __init__(self,weights=[1.0 / 2, 1.0],layer_indexs=[5, 22]):
        super(PerceptualLoss, self).__init__()
        self.criterion = nn.L1Loss().to(gpu_device)
        self.weights = weights
        self.layer_indexs = layer_indexs
        self.vgg = VGG(self.layer_indexs)
        self.vgg.features.load_state_dict(vgg19(pretrained=True).features.state_dict())
        self.vgg.to(gpu_device)
        self.vgg.eval()

        for parm in self.vgg.parameters():
            parm.requires_grad = False

    def forward(self, yPred, yGT):
        yPred = yPred.to(gpu_device)
        yGT = yGT.to(gpu_device)
        yPred_vgg, yGT_vgg = self.vgg(yPred), self.vgg(yGT)
        loss = 0
        for i in range(len(yPred_vgg)):
            loss += self.weights[i] * intensity_loss(yPred_vgg[i], yGT_vgg[i],l_num=2)
        return loss


def intensity_loss(gen_frames, gt_frames, l_num):
    return torch.mean(torch.abs((gen_frames - gt_frames) ** l_num))


# intra-grid constraint
def intra_grid_loss(pts):
    batch_size = pts.shape[0]

    delta_x = pts[:, :, 0:grid_w, 0] - pts[:, :, 1:grid_w + 1, 0]
    delta_y = pts[:, 0:grid_h, :, 1] - pts[:, 1:grid_h + 1, :, 1]

    loss_x = F.relu(delta_x + min_w)
    loss_y = F.relu(delta_y + min_h)

    loss = torch.mean(loss_x) + torch.mean(loss_y)
    return loss


# inter-grid constraint
def inter_grid_loss(train_mesh):
    w_edges = train_mesh[:, :, 0:grid_w, :] - train_mesh[:, :, 1:grid_w + 1, :]
    cos_w = torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 1:grid_w, :], 3) / \
            (torch.sqrt(torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 0:grid_w - 1, :], 3))
             * torch.sqrt(torch.sum(w_edges[:, :, 1:grid_w, :] * w_edges[:, :, 1:grid_w, :], 3)))
    # print("cos_w.shape")
    # print(cos_w.shape)
    delta_w_angle = 1 - cos_w

    h_edges = train_mesh[:, 0:grid_h, :, :] - train_mesh[:, 1:grid_h + 1, :, :]
    cos_h = torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 1:grid_h, :, :], 3) / \
            (torch.sqrt(torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 0:grid_h - 1, :, :], 3))
             * torch.sqrt(torch.sum(h_edges[:, 1:grid_h, :, :] * h_edges[:, 1:grid_h, :, :], 3)))
    delta_h_angle = 1 - cos_h

    loss = torch.mean(delta_w_angle) + torch.mean(delta_h_angle)
    return loss


def intensity_weight_loss(gen_frames, gt_frames, weight, l_num=1):
    # print(gen_frames.shape, gt_frames.shape)
    return torch.mean((torch.abs((gen_frames - gt_frames) * weight )** l_num))




# L2损失
def L2_loss(gen_frames, gt_frames, l_num=2):
    return torch.mean((torch.abs(gen_frames - gt_frames) ** l_num))

# 均方误差（MSE）损失
def mse_loss(flow, final_flow):
    return torch.mean((flow - final_flow) ** 2)

# 端点误差（End - Point Error，EPE）损失
def epe_loss(flow, final_flow):
    return torch.mean(torch.sqrt((flow[..., 0]-final_flow[..., 0]) ** 2+(flow[..., 1]-final_flow[..., 1]) ** 2))


def switch_load_balancing_loss(router_logits: torch.Tensor, top_k: int, num_experts: int) -> torch.Tensor:
    """
    改进后的 Switch Transformers 负载均衡损失

    Args:
        router_logits: 路由器的 logits，形状 [batch_size, num_experts]
        top_k: 路由到专家的数量
        num_experts: 专家总数

    Returns:
        total_loss: 总损失 = auxiliary_loss + z_loss
    """
    if top_k > num_experts:
        raise ValueError("top_k cannot be greater than num_experts.")

    # 1. 计算每个专家的路由概率
    router_probs = torch.softmax(router_logits, dim=-1)

    # 2. 找到每个样本被路由到的 top-k 专家
    # torch.topk 返回值：values, indices
    # 这里我们只关心被选中的专家 indices
    _, selected_experts = torch.topk(router_probs, k=top_k, dim=-1)

    # 3. 计算每个专家的实际负载（`f_i`）
    # 创建一个 one-hot 掩码，表示哪些样本被路由到了哪个专家
    one_hot_mask = F.one_hot(selected_experts, num_classes=num_experts).float()

    # 实际负载 f_i: 每个专家被选中的平均次数
    # 形状 [num_experts]，表示每个专家被选中作为 top-k 的比例
    actual_load = one_hot_mask.mean(dim=[0, 1])

    # 4. 计算每个专家的平均路由概率（`p_i`）
    # 形状 [num_experts]，表示每个专家被选中的平均概率
    avg_router_probs = router_probs.mean(dim=0)

    # 5. 计算辅助损失 (auxiliary loss)
    # 惩罚实际负载与路由概率平均值的差异
    # 理想情况下，每个专家的实际负载和平均路由概率都应该接近 1/num_experts
    # aux_loss = num_experts * sum(f_i * p_i)
    aux_loss = num_experts * torch.sum(actual_load * avg_router_probs)

    # 6. 计算 Z-损失 (z-loss)
    # 惩罚过大的 logits，防止训练不稳定
    z_loss = torch.mean(torch.square(router_logits))
    z_loss_weight = 0.001

    # 7. 计算总损失
    total_loss = aux_loss + z_loss * z_loss_weight

    return total_loss


def get_weight(n, start_weight=0.1, end_weight=0.01, total_rounds=50):
    """
    根据当前训练轮数n计算权重，权重从start_weight递减到end_weight。
    返回：当前训练轮数n的权重（保留两位小数）
    """
    # 计算每轮的递减量
    decrement_per_round = (start_weight - end_weight) / total_rounds

    # 计算当前轮数n对应的权重
    current_weight = start_weight - decrement_per_round * n

    # 保留两位小数
    weight = round(current_weight, 3)

    return weight


def get_weight_mask(mask, gt, pred, weight=10):
    mask = (mask * (weight - 1)) + 1
    gt = gt.mul(mask)
    pred = pred.mul(mask)
    return gt, pred

def adjust_weight(epoch, total_epoch, weight):
    return (1 - 0.9 * (epoch / total_epoch)) * weight



def l_num_loss(img1, img2, l_num=1):
    return torch.mean((torch.abs(img1 - img2) ** l_num))


def mask_flow_loss(flow, gt, task_ids, target_id=5):
    batch_size = flow.size(0)
    target_ids = torch.full((batch_size, 1), target_id, dtype=task_ids.dtype, device=task_ids.device)
    mask = task_ids == target_ids.squeeze(-1)
    flow_mask = flow * mask.view(batch_size, 1, 1, 1)
    gt_mask = gt * mask.view(batch_size, 1, 1, 1)

    return l_num_loss(flow_mask, gt_mask, 2)



def cal_task_cls(pre, gt):
    criterion = nn.CrossEntropyLoss()
    return criterion(pre, gt.long())



class IW_MOE_Total_Loss(nn.Module):
    def __init__(self, lam_appearance, group_A_tasks, group_B_tasks, target_perf_ssim, target_perf_psnr):
        super().__init__()
        self.lam_appearance = lam_appearance
        self.lam_ssim = 5e-6
        self.lam_primary_weight = 2
        self.lam_distill_weight = 0.01
        self.group_A_tasks = group_A_tasks
        self.group_B_tasks = group_B_tasks
        self.group_A_weight = 5.12
        self.group_B_weight = 2.56
        self.group_C_tasks = {1, 3}

        self.expert_number = 10
        self.residual_experts_number = 3
        self.top_k = 5
        self.residual_top_k = 1
        self.expert_weight = 0.01

        weights = [1.0]
        layer_indexs = [22]
        self.perceptual_loss = PerceptualLoss(weights=weights, layer_indexs=layer_indexs).to(gpu_device)
        self.grad_loss = GradLoss().to(gpu_device)
        self.l1_loss = nn.L1Loss().to(gpu_device)

        # 任务相关参数
        self.num_tasks = 5
        self.target_perf_ssim = torch.tensor(target_perf_ssim).cuda()  # 各任务目标SSIM值
        self.target_perf_psnr = torch.tensor(target_perf_psnr).cuda()
        self.max_perf_gap_id = torch.tensor(1, dtype=torch.float32).cuda()  # 标量值1.0

        self.active_residual_tasks = set()
        self.ssim_weight = 0.5
        self.psnr_weight = 0.5


    def set_active_residual_tasks(self, hard_task_ids):
        self.active_residual_tasks = set(hard_task_ids)

    def reset_active_residual_tasks(self):
        """Reset the set of active residual tasks"""
        self.active_residual_tasks = set()


    def calculate_performance_gap_by_group(self, avg_ssim, avg_psnr):

        # Reset the set of active residual tasks
        self.reset_active_residual_tasks()

        # Calculate the normalized performance gap for each task
        ssim_gaps = torch.abs(self.target_perf_ssim.cuda() - avg_ssim.cuda()) / (self.target_perf_ssim.cuda() + 1e-6)
        psnr_gaps = torch.abs(self.target_perf_psnr.cuda() - avg_psnr.cuda()) / (self.target_perf_psnr.cuda() + 1e-6)

        # Calculate the comprehensive difficulty score for each task
        combined_gaps = self.ssim_weight * ssim_gaps + self.psnr_weight * psnr_gaps

        # Calculate the average comprehensive difficulty of each task group
        group_A_combined_gap = torch.mean(torch.stack([combined_gaps[i] for i in self.group_A_tasks])) + self.group_A_weight
        group_B_combined_gap = torch.mean(torch.stack([combined_gaps[i] for i in self.group_B_tasks])) + self.group_B_weight

        # Identify the most challenging task group and determine the groups that were not selected
        if group_A_combined_gap > group_B_combined_gap:
            hardest_group = self.group_A_tasks
            other_group = self.group_B_tasks
        else:
            hardest_group = self.group_B_tasks
            other_group = self.group_A_tasks

        # Extract the comprehensive difficulty scores of all tasks from the unselected group
        other_group_gaps = torch.stack([combined_gaps[i] for i in other_group])

        # Find the biggest gap and corresponding index
        max_gap_in_other_group, local_hardest_index = torch.max(other_group_gaps, dim=0)


        other_group_list = sorted(list(other_group))
        hardest_task_in_other_group = other_group_list[local_hardest_index.item()]



        print('The most difficult task group:',hardest_group)
        print('The most difficult task ID not selected in the group:',hardest_task_in_other_group)

        result_tasks = hardest_group.copy()
        result_tasks.add(hardest_task_in_other_group)
        print('The most difficult task ID set:', self.group_C_tasks)

        return self.group_C_tasks


    def forward(self, flow, image_final, router_logits, resdiue_router_logits,point_cls, ds_flow, img_gt, task_id_tensor,epoch):
        batch_size = img_gt.size(0)
        appearance_loss_list = []
        perception_loss_list = []
        ds_loss_list = []
        independent_ds_losses = []

        for i in range(batch_size):
            j = task_id_tensor[i].item()

            if j in self.group_A_tasks:
                # print('group_A_tasks', j)
                restored_flow = flow[i].unsqueeze(0)[:, :, 64:64 + 256, 128:128 + 256]
                restored_image_final = image_final[i].unsqueeze(0)[:, :, 64:64 + 256, 128:128 + 256]
                restored_ds_flow = ds_flow[i].unsqueeze(0)[:, :, 64:64 + 256, 128:128 + 256]
                restored_img_gt = img_gt[i].unsqueeze(0)[:, :, 64:64 + 256, 128:128 + 256]
                # appearance loss
                appearance_loss = self.l1_loss(restored_image_final, restored_img_gt)
                # perception loss
                perception_loss = self.perceptual_loss(restored_image_final * 255., restored_img_gt * 255.)
                # ds loss
                ds_loss = L2_loss(restored_flow, restored_ds_flow)

                appearance_loss_list.append(appearance_loss)
                perception_loss_list.append(perception_loss)
                ds_loss_list.append(ds_loss)

            elif j in self.group_B_tasks:
                # print('group_B_tasks', j)
                # appearance loss
                appearance_loss = self.l1_loss(image_final[i], img_gt[i])
                # perception loss
                perception_loss = self.perceptual_loss(image_final[i] * 255., img_gt[i] * 255.)
                # ds loss
                ds_loss = L2_loss(flow[i], ds_flow[i])

                appearance_loss_list.append(appearance_loss)
                perception_loss_list.append(perception_loss)
                ds_loss_list.append(ds_loss)
            else:
                # 处理不在任何组中的任务（如果有的话）
                continue

        stacked_appearance_loss = torch.stack(appearance_loss_list)
        stacked_perception_loss = torch.stack(perception_loss_list)
        stacked_ds_loss = torch.stack(ds_loss_list)

        stacked_appearance_loss = torch.mean(stacked_appearance_loss)
        stacked_perception_loss = torch.mean(stacked_perception_loss)
        stacked_ds_loss = torch.mean(stacked_ds_loss)


        # aux loss
        if resdiue_router_logits is None:
            aux_loss = switch_load_balancing_loss(router_logits, self.top_k, self.expert_number)
        else:
            aux_loss = switch_load_balancing_loss(router_logits,self.top_k, self.expert_number) + switch_load_balancing_loss(resdiue_router_logits,self.residual_top_k, self.residual_experts_number)

        task_cls = cal_task_cls(point_cls, task_id_tensor) * 0.1


        if (epoch < 10):
            # total loss
            primary_img_loss = stacked_appearance_loss * self.lam_appearance + stacked_perception_loss * self.lam_ssim  # + mask_loss * self.lam_appearance
            total_loss = primary_img_loss * self.lam_primary_weight + aux_loss * self.expert_weight + task_cls
        else:
            # total loss
            primary_img_loss = stacked_appearance_loss * self.lam_appearance + stacked_perception_loss * self.lam_ssim  # + mask_loss * self.lam_appearance
            total_loss = primary_img_loss * self.lam_primary_weight + stacked_ds_loss * self.lam_distill_weight + aux_loss * self.expert_weight + task_cls


        return total_loss * 10, primary_img_loss * self.lam_primary_weight * 10, stacked_ds_loss * self.lam_distill_weight * 10, task_cls * 10







class Full_IW_MOE_Total_Loss(nn.Module):
    def __init__(self, lam_appearance, group_A_tasks, group_B_tasks, target_perf_ssim, target_perf_psnr):
        super().__init__()
        self.lam_appearance = lam_appearance
        self.lam_ssim = 5e-6
        self.lam_primary_weight = 2
        self.lam_distill_weight = 0.01
        self.group_A_tasks = {0,1,2}
        self.group_B_tasks = {3,4}
        self.group_A_weight = 5.12
        self.group_B_weight = 2.56
        self.group_ok_tasks = {1, 3}

        self.tasks_256 = group_A_tasks
        self.tasks_512 = group_B_tasks

        self.expert_number = 12
        self.residual_experts_number = 3
        self.top_k = 6
        self.residual_top_k = 1
        self.expert_weight = 0.01

        weights = [1.0]
        layer_indexs = [22]
        self.perceptual_loss = PerceptualLoss(weights=weights, layer_indexs=layer_indexs).to(gpu_device)
        self.grad_loss = GradLoss().to(gpu_device)
        self.l1_loss = nn.L1Loss().to(gpu_device)

        # 任务相关参数
        self.num_tasks = 5
        self.target_perf_ssim = torch.tensor(target_perf_ssim).cuda()  # 各任务目标SSIM值
        self.target_perf_psnr = torch.tensor(target_perf_psnr).cuda()
        self.max_perf_gap_id = torch.tensor(1, dtype=torch.float32).cuda()  # 标量值1.0

        self.active_residual_tasks = set()
        self.ssim_weight = 0.5
        self.psnr_weight = 0.5


    def set_active_residual_tasks(self, hard_task_ids):
        self.active_residual_tasks = set(hard_task_ids)

    def reset_active_residual_tasks(self):
        """Reset the set of active residual tasks"""
        self.active_residual_tasks = set()


    def calculate_performance_gap_by_group(self, avg_ssim, avg_psnr):

        # Reset the set of active residual tasks
        self.reset_active_residual_tasks()

        # Calculate the normalized performance gap for each task
        ssim_gaps = torch.abs(self.target_perf_ssim.cuda() - avg_ssim.cuda()) / (self.target_perf_ssim.cuda() + 1e-6)
        psnr_gaps = torch.abs(self.target_perf_psnr.cuda() - avg_psnr.cuda()) / (self.target_perf_psnr.cuda() + 1e-6)

        # Calculate the comprehensive difficulty score for each task
        combined_gaps = self.ssim_weight * ssim_gaps + self.psnr_weight * psnr_gaps

        # Calculate the average comprehensive difficulty of each task group
        group_A_combined_gap = torch.mean(torch.stack([combined_gaps[i] for i in self.group_A_tasks])) + self.group_A_weight
        group_B_combined_gap = torch.mean(torch.stack([combined_gaps[i] for i in self.group_B_tasks])) + self.group_B_weight

        # Identify the most challenging task group and determine the groups that were not selected
        if group_A_combined_gap > group_B_combined_gap:
            hardest_group = self.group_A_tasks
            other_group = self.group_B_tasks
        else:
            hardest_group = self.group_B_tasks
            other_group = self.group_A_tasks

        # Extract the comprehensive difficulty scores of all tasks from the unselected group
        other_group_gaps = torch.stack([combined_gaps[i] for i in other_group])

        # Find the biggest gap and corresponding index
        max_gap_in_other_group, local_hardest_index = torch.max(other_group_gaps, dim=0)


        other_group_list = sorted(list(other_group))
        hardest_task_in_other_group = other_group_list[local_hardest_index.item()]



        print('The most difficult task group:',hardest_group)
        print('The most difficult task ID not selected in the group:',hardest_task_in_other_group)

        result_tasks = hardest_group.copy()
        result_tasks.add(hardest_task_in_other_group)
        print('The most difficult task ID set:', self.group_ok_tasks)

        return self.group_ok_tasks



    def forward(self, flow, image_final, router_logits, resdiue_router_logits, point_cls, ds_flow, img_gt,
                    task_id_tensor, face_mask, face_weight, epoch):
        batch_size = img_gt.size(0)
        appearance_loss_list = []
        perception_loss_list = []
        ds_loss_list = []
        independent_ds_losses = []

        for i in range(batch_size):
            j = task_id_tensor[i].item()

            if j in self.tasks_256:
                # print('group_A_tasks', j)
                restored_flow = flow[i].unsqueeze(0)[:, :, 64:64 + 256, 128:128 + 256]
                restored_image_final = image_final[i].unsqueeze(0)[:, :, 64:64 + 256, 128:128 + 256]
                restored_ds_flow = ds_flow[i].unsqueeze(0)[:, :, 64:64 + 256, 128:128 + 256]
                restored_img_gt = img_gt[i].unsqueeze(0)[:, :, 64:64 + 256, 128:128 + 256]
                # appearance loss
                appearance_loss = self.l1_loss(restored_image_final, restored_img_gt)
                # perception loss
                perception_loss = self.perceptual_loss(restored_image_final * 255., restored_img_gt * 255.)
                # ds loss
                ds_loss = L2_loss(restored_flow, restored_ds_flow)

                appearance_loss_list.append(appearance_loss)
                perception_loss_list.append(perception_loss)
                ds_loss_list.append(ds_loss)

            elif j in self.tasks_512:
                # print('group_B_tasks', j)
                # appearance loss
                appearance_loss = self.l1_loss(image_final[i], img_gt[i])
                # perception loss
                perception_loss = self.perceptual_loss(image_final[i] * 255., img_gt[i] * 255.)
                # ds loss
                ds_loss = L2_loss(flow[i], ds_flow[i])

                appearance_loss_list.append(appearance_loss)
                perception_loss_list.append(perception_loss)
                ds_loss_list.append(ds_loss)
            else:
                # 处理不在任何组中的任务（如果有的话）
                continue

        stacked_appearance_loss = torch.stack(appearance_loss_list)
        stacked_perception_loss = torch.stack(perception_loss_list)
        stacked_ds_loss = torch.stack(ds_loss_list)

        stacked_appearance_loss = torch.mean(stacked_appearance_loss)
        stacked_perception_loss = torch.mean(stacked_perception_loss)
        stacked_ds_loss = torch.mean(stacked_ds_loss)


        # aux loss
        if resdiue_router_logits is None:
            aux_loss = switch_load_balancing_loss(router_logits, self.top_k, self.expert_number)
        else:
            aux_loss = switch_load_balancing_loss(router_logits,self.top_k, self.expert_number) + switch_load_balancing_loss(resdiue_router_logits,self.residual_top_k, self.residual_experts_number)

        task_cls = cal_task_cls(point_cls, task_id_tensor) * 0.1


        if (epoch < 10):
            # total loss
            primary_img_loss = stacked_appearance_loss * self.lam_appearance + stacked_perception_loss * self.lam_ssim  # + mask_loss * self.lam_appearance
            total_loss = primary_img_loss * self.lam_primary_weight + aux_loss * self.expert_weight + task_cls
        else:
            # total loss
            primary_img_loss = stacked_appearance_loss * self.lam_appearance + stacked_perception_loss * self.lam_ssim  # + mask_loss * self.lam_appearance
            total_loss = primary_img_loss * self.lam_primary_weight + stacked_ds_loss * self.lam_distill_weight + aux_loss * self.expert_weight + task_cls


        return total_loss * 10, primary_img_loss * self.lam_primary_weight * 10,stacked_ds_loss, task_cls * 10


