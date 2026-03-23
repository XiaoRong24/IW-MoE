import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import time
from tqdm import tqdm
import numpy as np
import random
import cv2
import json
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.models as models

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from net.IW_Moe_Model1 import IWMoeNetwork
from net.loss_functions import IW_MOE_Total_Loss
from net.loss import *

from utils.dataSet import IWTrainDataset, IWTestDataSet
from utils.learningRateScheduler import warmUpLearningRate
from utils.tp_grad import TaskPCGrad
from torch.utils.tensorboard import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler




# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary')
writer = SummaryWriter(log_dir=SUMMARY_DIR)

if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

setup_seed(2023)


def get_resolution_based_groups(unified_sizes):

    group_A_tasks = set()
    group_B_tasks = set()
    for task_id, size in enumerate(unified_sizes):
        if size == (256, 256):
            group_A_tasks.add(task_id)
        elif size == (512, 384):
            group_B_tasks.add(task_id)


    return group_A_tasks, group_B_tasks



def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)




import pandas as pd
import os
import csv


def val_once(net, epoch, epochs, criterion, optimizer):
    net.eval()

    test_num = len(test_loader_list)
    print('test_num:', test_num)

    # 根据实际任务数量动态创建数组
    num_tasks = len(test_loader_list)  # 使用数据中实际的任务数量
    print(f'Actual number of tasks in data: {num_tasks}')

    ssim_list = []
    psnr_list = []
    task_ssim = torch.zeros(num_tasks).cuda()
    task_psnr = torch.zeros(num_tasks).cuda()
    task_counts = torch.zeros(num_tasks).cuda()
    avg_ssim = torch.zeros(num_tasks).cuda()
    avg_psnr = torch.zeros(num_tasks).cuda()

    # 创建基础保存路径
    base_save_path = "../result/Final_Image"
    FillImage_save_path = "../result/FillImage_Image"
    flow_save_path = "../result/Flow_Image"
    metrics_save_path = "../result/Metrics"  # 新增：指标保存路径

    # 创建指标保存文件夹
    os.makedirs(metrics_save_path, exist_ok=True)

    for index in range(test_num):
        print("Task ID:", index)
        # 为当前任务创建专属文件夹
        # Final_Image
        task_save_path = os.path.join(base_save_path, f"task_{index}")
        os.makedirs(task_save_path, exist_ok=True)

        # FillImage
        FillImage_path = os.path.join(FillImage_save_path, f"task_{index}")
        os.makedirs(FillImage_path, exist_ok=True)

        # flow
        flow_path = os.path.join(flow_save_path, f"task_{index}")
        os.makedirs(flow_path, exist_ok=True)

        # 为当前任务创建CSV文件路径
        csv_file_path = os.path.join(metrics_save_path, f"task_{index}_metrics.csv")

        # 创建CSV文件并写入表头
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['image_index', 'psnr', 'ssim']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        test_each_data_batch = len(test_loader_list[index])
        with tqdm(total=test_each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict,
                  mininterval=0.3) as t_pbar:
            each_batch_psnr = 0
            each_batch_ssim = 0
            print("Start Test")

            with torch.no_grad():
                # 加载当前任务的数据
                test_loader = test_loader_list[index]
                for i, batch_value in enumerate(test_loader):
                    input_tensor = batch_value[0].float().cuda(device=args.device_ids[0])
                    gt_tensor = batch_value[1].float().cuda(device=args.device_ids[0])
                    mask_tensor = batch_value[2].float().cuda(device=args.device_ids[0])
                    task_id_tensor = batch_value[3].float().cuda(device=args.device_ids[0])

                    optimizer.zero_grad()
                    flow, final_image, router_logits, resdiue_router_logits ,task_cls = net.forward(input_tensor, mask_tensor,criterion.active_residual_tasks, epoch)

                    if index == 0 or index == 1 or index == 2:
                        final_image = final_image[:, :, 64:64 + 256, 128:128 + 256]
                        gt_tensor = gt_tensor[:, :, 64:64 + 256, 128:128 + 256]
                        flow = flow[:, :, 64:64 + 256, 128:128 + 256]

                    I1 = final_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                    I2 = gt_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

                    # 保存到对应任务的文件夹中
                    image_save_path = os.path.join(task_save_path, f"{i + 1:05d}.jpg")
                    cv2.imwrite(image_save_path, I1 * 255.)

                    FillImage = input_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                    image_save_path = os.path.join(FillImage_path, f"{i + 1:05d}.jpg")
                    cv2.imwrite(image_save_path, FillImage * 255.)

                    FlowImage = (flow[0]).cpu().detach().numpy().transpose(1, 2, 0)
                    FlowImage = flow_to_image(FlowImage)
                    image_save_path = os.path.join(flow_path, f"{i + 1:05d}.jpg")
                    cv2.imwrite(image_save_path, FlowImage)

                    psnr = compare_psnr(I1, I2, data_range=1)
                    ssim = compare_ssim(I1, I2, data_range=1, channel_axis=2)

                    # 将每张图片的指标追加到CSV文件
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({
                            'image_index': i + 1,
                            'psnr': psnr,
                            'ssim': ssim
                        })

                    task_ssim[index] += ssim
                    task_psnr[index] += psnr
                    task_counts[index] += 1

                    ssim_list.append(ssim)
                    psnr_list.append(psnr)

                    each_batch_psnr += psnr / args.test_batch_size
                    each_batch_ssim += ssim / args.test_batch_size
                    t_pbar.set_postfix({'average psnr': each_batch_psnr / (i + 1),
                                        'average ssim': each_batch_ssim / (i + 1)})
                    t_pbar.update(1)

    # 更新损失函数中的任务性能
    for task_id in range(criterion.num_tasks):
        if task_counts[task_id] > 0:
            avg_ssim[task_id] = task_ssim[task_id] / task_counts[task_id]
            avg_psnr[task_id] = task_psnr[task_id] / task_counts[task_id]
            print("Task ID:", task_id)
            print("SSIM:", avg_ssim[task_id])
            print('PSNR:', avg_psnr[task_id])

            # 在CSV文件末尾添加平均指标（可选）
            csv_file_path = os.path.join(metrics_save_path, f"task_{task_id}_metrics.csv")
            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({})  # 空行分隔
                writer.writerow({
                    'image_index': 'Average',
                    'psnr': avg_psnr[task_id].item(),
                    'ssim': avg_ssim[task_id].item()
                })

    print("Average SSIM:", avg_ssim)

    return avg_ssim, avg_psnr







def train(net,saveModelName,criterion,optimizer,scheduler,start_epochs=0, end_epochs=1):

    loss_history = []
    switch_epoch = 50
    ssim_temp = 0

    epoch = args.resume_epoch

    # 开始计时
    start_time = time.time()  # 记录训练开始时间

    # testing
    avg_ssim_per_task, avg_psnr_per_task = val_once(net, epoch, end_epochs, criterion, optimizer)

    # 结束计时
    end_time = time.time()  # 记录训练结束时间

    # 计算训练时长
    elapsed_time = end_time - start_time  # 训练用时（秒）
    # 转换为时分秒格式
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    # 输出训练时间
    print(f"Training Time: {hours:02}:{minutes:02}:{seconds:02}")


if __name__ == "__main__":

    print('<==================== setting arguments ===================>\n')

    parser = argparse.ArgumentParser()
    '''Implementation details'''
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--end_epochs', type=int, default=201)
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('-w', '--warmup', type=bool, default=True)
    parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')
    parser.add_argument("--eta_min", type=float, default=1e-6, help="final learning rate")
    # 各任务目标性能
    parser.add_argument('--target_perf_ssim', type=list, default=[0.5450, 0.7488, 0.7711, 0.6286, 0.7952])
    parser.add_argument('--target_perf_psnr', type=list, default=[18.3522, 22.3443, 21.7201, 21.8102, 22.5402])
    parser.add_argument('--unified_size', type=list, default=[(256, 256), (256, 256), (256, 256), (512, 384), (512, 384)])


    '''Network details'''
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='../model/distill_model')
    parser.add_argument('--lam_perception', type=float, default=0.2)
    parser.add_argument('--lam_mask', type=float, default=1)
    parser.add_argument('--lam_mesh', type=float, default=1)
    parser.add_argument('--lam_appearance', type=float, default=1)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--resume_epoch', type=int, default=90)

    '''Dataset settings'''
    parser.add_argument('--train_path', type=str,
                        default=['../../Dataset/WideAngle_Image/training/', '../../Dataset/Fisheye_Image/training/',
                                 '../../Dataset/UnrollingShutter/training/', '../../Dataset/DRC-D/training/',
                                 '../../Dataset/DIR-D/training/'])
    parser.add_argument('--test_path', type=str,
                        default=['../../Dataset/WideAngle_Image/testing/', '../../Dataset/Fisheye_Image/testing/',
                                 '../../Dataset/UnrollingShutter/testing/', '../../Dataset/DRC-D/testing/',
                                 '../../Dataset/DIR-D/testing/'])


    args = parser.parse_args()
    print(args)

    group_A_tasks, group_B_tasks = get_resolution_based_groups(args.unified_size)

    print('Group A tasks:', group_A_tasks)  # {0, 1, 2}
    print('Group B tasks:', group_B_tasks)  # {3, 4}


    # Load Testing Data
    test_loader_list = [DataLoader(dataset=IWTestDataSet(test_path, i, args.unified_size[i], group_A_tasks, group_B_tasks), batch_size=args.test_batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False) \
                        for i, test_path in enumerate(args.test_path)]
    for i, test_path in enumerate(args.test_path):
        print(f'Task ID:{i}对应任务:{test_path}')




    # define Loss
    criterion = IW_MOE_Total_Loss(args.lam_appearance, group_A_tasks, group_B_tasks, args.target_perf_ssim, args.target_perf_psnr).cuda(device=args.device_ids[0])
    net = IWMoeNetwork()
    net = net.to(device=args.device_ids[0])


    base_optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    optimizer = TaskPCGrad(base_optimizer)

    # 动混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    lrScheduler = warmUpLearningRate(args.end_epochs, warm_up_epochs=10, scheduler='cosine')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lr_lambda=lrScheduler)

    if args.resume:
        for i in range(0, (args.start_epochs + 1)):
            scheduler.step()
        args.start_epochs = args.resume_epoch
        load_path = args.save_model_name + "_" + "epoch" + str(args.resume_epoch) + ".pkl"
        checkpoint = torch.load(load_path, map_location='cpu')

        # 获取当前模型的状态字典
        model_dict = net.state_dict()

        # 过滤掉不匹配的层 (例如，如果新加的 new_task_expert 不在预训练模型中)
        state_dict_filtered = {k: v for k, v in checkpoint['net_state_dict'].items()
                               if k in model_dict and model_dict[k].shape == v.shape}

        # 更新当前模型的状态字典
        model_dict.update(state_dict_filtered)

        # 加载过滤后的状态字典到模型
        net.load_state_dict(model_dict, strict=False)

        # 恢复 active_residual_tasks
        criterion.set_active_residual_tasks(set(checkpoint['active_residual_tasks']))
        print("-----resumed train, loaded model state dict success------")



    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    # start train
    train(net, args.save_model_name, criterion, optimizer, scheduler, args.start_epochs, args.end_epochs)







