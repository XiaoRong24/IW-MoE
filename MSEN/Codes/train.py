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

from net.IW_Moe_Model import IWMoeNetwork
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


def train_once(net,each_data_batch,epoch,epochs,criterion, optimizer):
    net.train()
    with tqdm(total=each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        each_batch_all_loss = 0
        each_batch_primary_loss = 0
        each_batch_ds_loss = 0
        each_batch_task_loss= 0

        projection_interval = 100

        print("Start Train")

        for i, batch_value in enumerate(train_loader):
            input_tesnor = batch_value[0].float().cuda(device=args.device_ids[0])
            gt_tesnor = batch_value[1].float().cuda(device=args.device_ids[0])
            mask_tensor = batch_value[2].float().cuda(device=args.device_ids[0])
            task_id_tensor = batch_value[3].float().cuda(device=args.device_ids[0])
            ds_flow = batch_value[4].float().cuda(device=args.device_ids[0])


            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                flow, final_image, router_logits, resdiue_router_logits, task_cls = net.forward(input_tesnor, mask_tensor,criterion.active_residual_tasks, epoch)

                total_loss, primary_img_loss, ds_loss, task_loss = criterion(flow, final_image, router_logits, resdiue_router_logits, task_cls, ds_flow, gt_tesnor, task_id_tensor,epoch)

                #  super_img_loss,

                # # 核心修改：基于频率的混合策略
                # if (i + 1) % projection_interval == 0:
                #     # ====== PCGrad 投影轮次 ======
                #
                #     # 1. 筛选出有效的 DS 损失
                #     # losses_to_project = [
                #     #     loss for loss in independent_ds_losses if loss.item() != 0
                #     # ]
                #
                #     # 2. 创建所有损失的列表，以便PCGrad处理
                #     all_losses_list =  [super_img_loss, primary_img_loss]
                #
                #     # 3. 将所有损失传递给PCGrad
                #     if all_losses_list:
                #         optimizer.zero_grad()
                #         # 1. 先用scaler缩放每个损失
                #         scaled_losses = [scaler.scale(loss) for loss in all_losses_list]
                #         optimizer.pc_backward(scaled_losses)
                # else:
                #     # ====== 常规反向传播轮次 ======
                #     # 对总损失进行反向传播，让所有损失都参与梯度计算
                #     scaler.scale(total_loss).backward()

            scaler.scale(total_loss).backward()
            # scaler.scale(total_loss).backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            scaler.step(optimizer.optimizer)
            scaler.update()

            # for param in net.meshRegression.parameters():
            #     if param.requires_grad and param.grad is not None:
            #         print(param.grad.norm())
                    # print(param.grad)




            each_batch_all_loss += total_loss.item() / args.train_batch_size
            each_batch_primary_loss += primary_img_loss.item() / args.train_batch_size
            each_batch_ds_loss += ds_loss.item() / args.train_batch_size
            each_batch_task_loss += task_loss.item() / args.train_batch_size
            pbar.set_postfix({'Lsum': each_batch_all_loss / (i + 1),
                              'Lpri': each_batch_primary_loss / (i + 1),
                              'Lds': each_batch_ds_loss / (i + 1),
                              'Ltask': each_batch_task_loss / (i + 1),
                              'lr': scheduler.get_last_lr()[0]})
            pbar.update(1)



            # 每100个batch记录一次指标
            if (i + 1) % 1000 == 0:
                # 计算两种网络的相对贡献度
                if hasattr(net.master_slave_moe,
                           'moe_output_buffer') and net.master_slave_moe.moe_output_buffer is not None:
                    moe_contribution = torch.mean(torch.abs(net.master_slave_moe.moe_output_buffer)).item()
                    shared_contribution = torch.mean(torch.abs(net.master_slave_moe.shared_output_buffer)).item()
                    total_contribution = moe_contribution + shared_contribution

                    # 记录 MoE 和 Shared 专家的贡献度比例
                    if total_contribution > 1e-6:  # 避免除以零
                        writer.add_scalar('Contribution/MoE_Ratio',
                                          moe_contribution / total_contribution,
                                          epoch * len(train_loader) + i)
                        writer.add_scalar('Contribution/Shared_Ratio',
                                          shared_contribution / total_contribution,
                                          epoch * len(train_loader) + i)
                        writer.add_scalar('Contribution/Total_Contribution',
                                          total_contribution,
                                          epoch * len(train_loader) + i)

                # 记录主路由（SparseMOE）的专家分布
                if router_logits is not None:
                    router_probs = torch.softmax(router_logits, dim=-1)
                    avg_router_probs = torch.mean(router_probs, dim=0).cpu().detach().numpy()

                    # 记录每个主专家的使用概率
                    for expert_idx in range(len(avg_router_probs)):
                        writer.add_scalar(f'Expert_Usage/Main_Expert_{expert_idx}',
                                          avg_router_probs[expert_idx],
                                          epoch * len(train_loader) + i)

                    # 记录专家使用分布的熵（衡量专家专业化程度）
                    expert_entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1).mean()
                    writer.add_scalar('Expert_Usage/Main_Expert_Entropy',
                                      expert_entropy.item(),
                                      epoch * len(train_loader) + i)

                    # 记录专家使用分布的稀疏性
                    expert_sparsity = torch.mean((router_probs > 0.1).float(), dim=-1).mean()
                    writer.add_scalar('Expert_Usage/Main_Expert_Sparsity',
                                      expert_sparsity.item(),
                                      epoch * len(train_loader) + i)

                # 记录残差路由（DynamicResidualMOE）的专家分布
                if resdiue_router_logits is not None:
                    resdiue_router_probs = torch.softmax(resdiue_router_logits, dim=-1)
                    avg_resdiue_router_probs = torch.mean(resdiue_router_probs, dim=0).cpu().detach().numpy()

                    # 记录每个残差专家的使用概率
                    for expert_idx in range(len(avg_resdiue_router_probs)):
                        writer.add_scalar(f'Expert_Usage/Residual_Expert_{expert_idx}',
                                          avg_resdiue_router_probs[expert_idx],
                                          epoch * len(train_loader) + i)

                    # 记录残差专家使用情况
                    residual_expert_entropy = -torch.sum(resdiue_router_probs * torch.log(resdiue_router_probs + 1e-8),
                                                         dim=-1).mean()
                    writer.add_scalar('Expert_Usage/Residual_Expert_Entropy',
                                      residual_expert_entropy.item(),
                                      epoch * len(train_loader) + i)


        print("\nFinish Train")
        return each_batch_all_loss / each_data_batch




def val_once(net,epoch,epochs,criterion,optimizer):
    net.eval()
    ssim_list = []
    psnr_list = []
    task_ssim = torch.zeros(criterion.num_tasks).cuda()
    task_psnr = torch.zeros(criterion.num_tasks).cuda()
    task_counts = torch.zeros(criterion.num_tasks).cuda()
    avg_ssim = torch.zeros(criterion.num_tasks).cuda()
    avg_psnr = torch.zeros(criterion.num_tasks).cuda()
    test_num = len(test_loader_list)
    print('test_num:', test_num)
    for index in range(test_num):
        print("Task ID:", index)
        test_each_data_batch = len(test_loader_list[index])
        with tqdm(total=test_each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as t_pbar:
            each_batch_psnr = 0
            each_batch_ssim = 0
            print("Start Test")
            with torch.no_grad():
                # 加载当前任务的数据。
                test_loader = test_loader_list[index]
                for i, batch_value in enumerate(test_loader):
                    input_tesnor = batch_value[0].float().cuda(device=args.device_ids[0])
                    gt_tesnor = batch_value[1].float().cuda(device=args.device_ids[0])
                    mask_tensor = batch_value[2].float().cuda(device=args.device_ids[0])
                    # 多任务学习中，用于区分不同的任务
                    task_id_tensor = batch_value[3].float().cuda(device=args.device_ids[0])

                    optimizer.zero_grad()
                    flow, final_image, router_logits, resdiue_router_logits ,task_cls = net.forward(input_tesnor, mask_tensor,criterion.active_residual_tasks, epoch)

                    if index == 0 or index == 1 or index == 2:
                        final_image = final_image[:, :, 64:64 + 256, 128:128 + 256]
                        gt_tesnor = gt_tesnor[:, :, 64:64 + 256, 128:128 + 256]

                    I1 = final_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                    I2 = gt_tesnor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

                    psnr = compare_psnr(I1, I2, data_range=1)
                    ssim = compare_ssim(I1, I2, data_range=1, channel_axis=2)

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

    print("Average SSIM:", avg_ssim)

    # 3. 计算性能差距并找到最困难的3个任务
    hardest_tasks_ids = criterion.calculate_performance_gap_by_group(avg_ssim, avg_psnr)

    # 4. 动态更新模型中的活跃残差任务集合
    criterion.set_active_residual_tasks(hardest_tasks_ids)
    # print(f"Epoch {epoch}: The 3 hardest tasks are {hardest_tasks_ids}. Activated residual experts for them.")

    #
    # print(f"Task Weights: {criterion.task_weights.cpu().numpy()}")
    print("\nFinish Test")

    return avg_ssim, avg_psnr



def train(net,saveModelName,criterion,optimizer,scheduler,start_epochs=0, end_epochs=1):

    loss_history = []
    switch_epoch = 50
    ssim_temp = 0

    train_each_data_batch = len(train_loader)

    for epoch in range(start_epochs,end_epochs):
        # 开始计时
        start_time = time.time()  # 记录训练开始时间
        # training
        each_batch_all_loss = train_once(net,train_each_data_batch,epoch, end_epochs,criterion,optimizer)

        # testing
        if epoch % 10 == 0:
            avg_ssim_per_task, avg_psnr_per_task = val_once(net, epoch, end_epochs, criterion, optimizer)
            # 将 SSIM 和 PSNR 张量转换为列表
            ssim_list = avg_ssim_per_task.cpu().tolist()
            psnr_list = avg_psnr_per_task.cpu().tolist()

            # 对列表中的每个浮点数进行四舍五入，保留小数点后四位
            rounded_ssim_list = [round(x, 4) for x in ssim_list]
            rounded_psnr_list = [round(x, 4) for x in psnr_list]

            # 准备要保存的数据
            epoch_metrics = {
                'epoch': epoch,
                'ssim_per_task': rounded_ssim_list,
                'psnr_per_task': rounded_psnr_list
            }
            # 以追加模式('a')打开文件，并写入JSON数据
            with open(saveModelName+ "_metrics" + ".jsonl", 'a') as f:
                json.dump(epoch_metrics, f)
                f.write('\n')

            print(f"Metrics for epoch {epoch} saved")

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


        # learning rate scheduler
        scheduler.step()
        loss_history.append(each_batch_all_loss)

        if (epoch + 1) % 10 ==0 or epoch >= int(end_epochs-5):
            checkpoint = {
                'epoch': epoch + 1,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'active_residual_tasks': list(criterion.active_residual_tasks),
            }
            torch.save(checkpoint, saveModelName + "_" + "epoch" + str(epoch + 1) + ".pkl")



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
    parser.add_argument('--resume_epoch', type=int, default=200)

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

    # Load Training Data
    train_dataset = IWTrainDataset(args.train_path, group_A_tasks, group_B_tasks)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

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

        # 加载优化器状态
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("-----resume train, load optimizer state dict success------")

        # 加载 scheduler 状态
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("-----resume train, load scheduler state dict success------")

    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    print('criterion.active_residual_tasks', criterion.active_residual_tasks)

    # start train
    train(net, args.save_model_name, criterion, optimizer, scheduler, args.start_epochs, args.end_epochs)







