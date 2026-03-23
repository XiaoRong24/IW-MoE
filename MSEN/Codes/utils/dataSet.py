import os
import glob
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import utils.constant as constant
from utils.utils_op import data_aug
from collections import OrderedDict
import random
# import torchvision.transforms.functional as F
import torch.nn.functional as F
from random import random

grid_w = constant.GRID_W
grid_h = constant.GRID_H

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


class IWTrainDataset(Dataset):
    def __init__(self, paths,group_A_tasks,group_B_tasks):

        self.width = 512
        self.height = 384
        self.prob = 0.5
        self.group_A_tasks = group_A_tasks
        self.group_B_tasks = group_B_tasks
        self.input_images = []
        self.gt_images = []
        self.masks = []
        self.flows = []
        self.task_id = []
        for index, path in enumerate(paths):
            inputs = glob.glob(os.path.join(path, 'input/', '*.*'))
            gts = glob.glob(os.path.join(path, 'gt/', '*.*'))
            masks = glob.glob(os.path.join(path, 'mask/', '*.*'))
            flows = glob.glob(os.path.join(path, 'distill_flow/', '*.*'))
            inputs.sort()
            gts.sort()
            masks.sort()
            flows.sort()


            lens = len(inputs)
            index_array = [index] * lens
            self.task_id.extend(index_array)
            self.input_images.extend(inputs)
            self.gt_images.extend(gts)
            self.masks.extend(masks)
            self.flows.extend(flows)

        print("total dataset num: ", len(self.input_images))

    def __getitem__(self, index):

        '''load images'''
        task_id = self.task_id[index]
        padding = (128, 128, 64, 64)  # 左右各128，上下各64

        if task_id in self.group_B_tasks:
            input_src = cv2.imread(self.input_images[index])
            input_resized = cv2.resize(input_src, (self.width, self.height))
            gt = cv2.imread(self.gt_images[index])
            gt = cv2.resize(gt, (self.width, self.height))
            input_resized, gt = data_aug(input_resized, gt)

            input_resized = input_resized.astype(dtype=np.float32)
            input_resized = input_resized / 255.0
            input_resized = np.transpose(input_resized, [2, 0, 1])
            gt = gt.astype(dtype=np.float32)
            gt = gt / 255.0
            gt = np.transpose(gt, [2, 0, 1])

            '''load mask'''
            mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.width, self.height))
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(dtype=np.float32)
            mask = mask / 255.0
            mask = np.transpose(mask, [2, 0, 1])
            mask_tensor = torch.tensor(mask)

            '''load flow'''
            flow = np.load(self.flows[index])
            flow = flow.astype(dtype=np.float32)
            flow_tensor = torch.tensor(flow)

            input_tensor = torch.tensor(input_resized)
            gt_tensor = torch.tensor(gt)
            task_id_tensor = torch.tensor(task_id, dtype=torch.int64)

        elif task_id in self.group_A_tasks:
            input_src = cv2.imread(self.input_images[index])
            gt = cv2.imread(self.gt_images[index])
            input_resized, gt = data_aug(input_src, gt)

            input_resized = input_resized.astype(dtype=np.float32)
            input_resized = input_resized / 255.0
            input_resized = np.transpose(input_resized, [2, 0, 1])
            gt = gt.astype(dtype=np.float32)
            gt = gt / 255.0
            gt = np.transpose(gt, [2, 0, 1])

            '''load mask'''
            mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(dtype=np.float32)
            mask = mask / 255.0
            mask = np.transpose(mask, [2, 0, 1])
            mask_tensor = torch.tensor(mask)

            '''load flow'''
            flow = np.load(self.flows[index])
            flow = flow.astype(dtype=np.float32)
            flow_tensor = torch.tensor(flow)

            input_tensor = torch.tensor(input_resized)
            gt_tensor = torch.tensor(gt)
            task_id_tensor = torch.tensor(task_id, dtype=torch.int64)

            flow_tensor = F.pad(flow_tensor, pad=padding, mode='constant', value=0)
            gt_tensor = F.pad(gt_tensor, pad=padding, mode='constant', value=0)
            input_tensor = F.pad(input_tensor, pad=padding, mode='constant', value=0)
            mask_tensor = F.pad(mask_tensor, pad=padding, mode='constant', value=0)


        return (input_tensor, gt_tensor, mask_tensor, task_id_tensor, flow_tensor)

    def __len__(self):
        return len(self.input_images)


class IWTestDataSet(Dataset):
    def __init__(self, data_path, task_id, unifide_size, group_A_tasks, group_B_tasks):
        self.width = unifide_size[0]
        self.height = unifide_size[1]
        self.test_path = data_path
        self.datas = OrderedDict()
        self.task_id = task_id
        self.group_A_tasks = group_A_tasks
        self.group_B_tasks = group_B_tasks


        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'gt' or data_name == 'mask':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.*'))
                self.datas[data_name]['image'].sort()

    def __getitem__(self, index):

        '''load images'''
        padding = (128, 128, 64, 64)  # 左右各128，上下各64
        if self.task_id in self.group_B_tasks:
            input = cv2.imread(self.datas['input']['image'][index])
            input = cv2.resize(input, (self.width, self.height))
            input = input.astype(dtype=np.float32)
            input = input / 255.0
            input = np.transpose(input, [2, 0, 1])

            gt = cv2.imread(self.datas['gt']['image'][index])
            gt = cv2.resize(gt, (self.width, self.height))
            gt = gt.astype(dtype=np.float32)
            gt = gt / 255.0
            gt = np.transpose(gt, [2, 0, 1])

            mask = cv2.imread(self.datas['mask']['image'][index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.width, self.height))
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(dtype=np.float32)
            mask = mask / 255.0
            mask = np.transpose(mask, [2, 0, 1])

            input_tensor = torch.tensor(input)
            gt_tensor = torch.tensor(gt)
            mask_tensor = torch.tensor(mask)
            task_id_tensor = torch.tensor(self.task_id, dtype=torch.int64)
        elif self.task_id in self.group_A_tasks:
            input = cv2.imread(self.datas['input']['image'][index])
            input = cv2.resize(input, (self.width, self.height))
            input = input.astype(dtype=np.float32)
            input = input / 255.0
            input = np.transpose(input, [2, 0, 1])

            gt = cv2.imread(self.datas['gt']['image'][index])
            gt = cv2.resize(gt, (self.width, self.height))
            gt = gt.astype(dtype=np.float32)
            gt = gt / 255.0
            gt = np.transpose(gt, [2, 0, 1])

            mask = cv2.imread(self.datas['mask']['image'][index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.width, self.height))
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(dtype=np.float32)
            mask = mask / 255.0
            mask = np.transpose(mask, [2, 0, 1])

            input_tensor = torch.tensor(input)
            gt_tensor = torch.tensor(gt)
            mask_tensor = torch.tensor(mask)
            task_id_tensor = torch.tensor(self.task_id, dtype=torch.int64)

            gt_tensor = F.pad(gt_tensor, pad=padding, mode='constant', value=0)
            input_tensor = F.pad(input_tensor, pad=padding, mode='constant', value=0)
            mask_tensor = F.pad(mask_tensor, pad=padding, mode='constant', value=0)

        file_name = os.path.basename(self.datas['input']['image'][index])
        file_name, _ = os.path.splitext(file_name)

        return (input_tensor, gt_tensor, mask_tensor, task_id_tensor)

    def __len__(self):
        return len(self.datas['input']['image'])



class FullIWTrainDataset(Dataset):
    def __init__(self, paths,group_A_tasks,group_B_tasks):

        self.width = 512
        self.height = 384
        self.prob = 0.5
        self.group_A_tasks = group_A_tasks
        self.group_B_tasks = group_B_tasks
        self.input_images = []
        self.gt_images = []
        self.masks = []
        self.flows = []
        self.task_id = []
        for index, path in enumerate(paths):
            inputs = glob.glob(os.path.join(path, 'input/', '*.*'))
            gts = glob.glob(os.path.join(path, 'gt/', '*.*'))
            masks = glob.glob(os.path.join(path, 'mask/', '*.*'))
            flows = glob.glob(os.path.join(path, 'distill_flow/', '*.*'))
            inputs.sort()
            gts.sort()
            masks.sort()
            flows.sort()


            lens = len(inputs)
            index_array = [index] * lens
            self.task_id.extend(index_array)
            self.input_images.extend(inputs)
            self.gt_images.extend(gts)
            self.masks.extend(masks)
            self.flows.extend(flows)

        print("total dataset num: ", len(self.input_images))

    def __getitem__(self, index):

        '''load images'''
        task_id = self.task_id[index]
        padding = (128, 128, 64, 64)  # 左右各128，上下各64

        if task_id in self.group_B_tasks:
            input_src = cv2.imread(self.input_images[index])
            input_resized = cv2.resize(input_src, (self.width, self.height))
            gt = cv2.imread(self.gt_images[index])
            gt = cv2.resize(gt, (self.width, self.height))
            input_resized, gt = data_aug(input_resized, gt)

            input_resized = input_resized.astype(dtype=np.float32)
            input_resized = input_resized / 255.0
            input_resized = np.transpose(input_resized, [2, 0, 1])
            gt = gt.astype(dtype=np.float32)
            gt = gt / 255.0
            gt = np.transpose(gt, [2, 0, 1])

            '''load mask'''
            mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.width, self.height))
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(dtype=np.float32)
            mask = mask / 255.0
            mask = np.transpose(mask, [2, 0, 1])
            mask_tensor = torch.tensor(mask)

            '''load flow'''
            flow = np.load(self.flows[index])
            flow = flow.astype(dtype=np.float32)
            flow_tensor = torch.tensor(flow)

            input_tensor = torch.tensor(input_resized)
            gt_tensor = torch.tensor(gt)
            task_id_tensor = torch.tensor(task_id, dtype=torch.int64)

        elif task_id in self.group_A_tasks:
            input_src = cv2.imread(self.input_images[index])
            gt = cv2.imread(self.gt_images[index])
            input_resized, gt = data_aug(input_src, gt)

            input_resized = input_resized.astype(dtype=np.float32)
            input_resized = input_resized / 255.0
            input_resized = np.transpose(input_resized, [2, 0, 1])
            gt = gt.astype(dtype=np.float32)
            gt = gt / 255.0
            gt = np.transpose(gt, [2, 0, 1])

            '''load mask'''
            mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(dtype=np.float32)
            mask = mask / 255.0
            mask = np.transpose(mask, [2, 0, 1])
            mask_tensor = torch.tensor(mask)

            '''load flow'''
            flow = np.load(self.flows[index])
            flow = flow.astype(dtype=np.float32)
            flow_tensor = torch.tensor(flow)

            input_tensor = torch.tensor(input_resized)
            gt_tensor = torch.tensor(gt)
            task_id_tensor = torch.tensor(task_id, dtype=torch.int64)

            flow_tensor = F.pad(flow_tensor, pad=padding, mode='constant', value=0)
            gt_tensor = F.pad(gt_tensor, pad=padding, mode='constant', value=0)
            input_tensor = F.pad(input_tensor, pad=padding, mode='constant', value=0)
            mask_tensor = F.pad(mask_tensor, pad=padding, mode='constant', value=0)

        '''load flow and face mask for the portrait task'''
        if (task_id == 5):
            face_mask_path = self.input_images[index].replace('/input/', '/mask_face/')
            facemask = cv2.imread(face_mask_path, 0)
            facemask = facemask.astype(dtype=np.float32)
            facemask = (facemask / 255.0)
            facemask = np.expand_dims(facemask, axis=-1)
            facemask = np.transpose(facemask, [2, 0, 1])
            face_mask = torch.tensor(facemask)
            mask_sum = torch.sum(face_mask)
            weight = self.width * self.height / mask_sum - 1
            weight = torch.max(weight / 3, torch.ones(1))
            face_weight = weight.unsqueeze(-1).unsqueeze(-1)
        else:
            face_mask = torch.zeros_like(mask_tensor)
            face_weight = torch.mean(face_mask).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return (input_tensor, gt_tensor, mask_tensor, task_id_tensor, flow_tensor, face_mask, face_weight)

    def __len__(self):
        return len(self.input_images)




class AdaptiveTestDataSet(Dataset):
    def __init__(self, data_path):
        self.target_w = 512  # 标准宽度
        self.target_h = 384  # 标准高度
        self.test_path = data_path
        self.datas = OrderedDict()


        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'mask':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.*'))
                self.datas[data_name]['image'].sort()


    def __getitem__(self, index):

        # 1. 加载原始图像 (不先进行 resize，保留原始比例)
        input_path = self.datas['input']['image'][index]
        mask_path = self.datas['mask']['image'][index]

        input = cv2.imread(input_path)
        input_h, input_w = input.shape[:2]
        if input_w > self.target_w or input_h > self.target_h:
            # print('input_w',input_w,'input_h',input_h)
            input = cv2.resize(input, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        input = input.astype(dtype=np.float32)
        input = input / 255.0
        input = np.transpose(input, [2, 0, 1])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_h, mask_w = mask.shape[:2]
        if mask_w > self.target_w or mask_h > self.target_h:
            mask = cv2.resize(mask, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(dtype=np.float32)
        mask = mask / 255.0
        mask = np.transpose(mask, [2, 0, 1])

        # 获取原始高度和宽度
        h_orig, w_orig = input.shape[1:3]
        # print('h_orig: {}, w_orig: {}'.format(h_orig, w_orig))


        input_tensor = torch.tensor(input)
        mask_tensor = torch.tensor(mask)


        # 3. 尺寸检查与填充逻辑
        # 判断：如果宽度 < 512 或 高度 < 384
        if w_orig < self.target_w or h_orig < self.target_h:
            # 计算需要补齐到 (512, 384) 的差值
            pad_w = int(max(0, self.target_w - w_orig) / 2)
            pad_h = int(max(0, self.target_h - h_orig) / 2)
            # print('pad_w', pad_w)
            # print('pad_h', pad_h)

            # 先将图像补齐到 512x384 (在右侧和下方填充)
            # F.pad 参数顺序: (左, 右, 上, 下)
            std_padding = (pad_w, pad_w, pad_h, pad_h)
            # print('std_padding', std_padding)
            input_tensor = F.pad(input_tensor, pad=std_padding, mode='constant', value=0)
            mask_tensor = F.pad(mask_tensor, pad=std_padding, mode='constant', value=0)

        # 4. 叠加你要求的固定填充 (左右128, 上下64)
        # extra_padding = (128, 128, 64, 64)
        # input_tensor = F.pad(input_tensor, pad=extra_padding, mode='constant', value=0)
        # mask_tensor = F.pad(mask_tensor, pad=extra_padding, mode='constant', value=0)

        # print('input_tensor', input_tensor.shape)

        # 5. 记录元数据用于测试阶段裁剪
        meta = {
            'orig_size': (h_orig, w_orig),
            'file_name': os.path.basename(input_path)
        }
        # print('input_tensor.shape', input_tensor.shape)

        return input_tensor, mask_tensor, meta

    def __len__(self):
        # 必须确保这里有闭合括号！
        return len(self.datas['input']['image'])





class Fine_Tuning_TrainDataset(Dataset):
    def __init__(self, paths):

        self.width = 512
        self.height = 384
        self.prob = 0.5
        self.input_images = []
        self.gt_images = []
        self.masks = []
        self.flows = []
        self.task_id = []
        for index, path in enumerate(paths):
            inputs = glob.glob(os.path.join(path, 'input/', '*.*'))
            gts = glob.glob(os.path.join(path, 'gt/', '*.*'))
            masks = glob.glob(os.path.join(path, 'mask/', '*.*'))
            flows = glob.glob(os.path.join(path, 'distill_flow/', '*.*'))
            inputs.sort()
            gts.sort()
            masks.sort()
            flows.sort()


            lens = len(inputs)
            index_array = [index] * lens
            self.task_id.extend(index_array)
            self.input_images.extend(inputs)
            self.gt_images.extend(gts)
            self.masks.extend(masks)
            self.flows.extend(flows)

        print("total dataset num: ", len(self.input_images))

    def __getitem__(self, index):

        '''load images'''
        task_id = self.task_id[index]
        padding = (128, 128, 64, 64)  # 左右各128，上下各64
        #
        # input_src = cv2.imread(self.input_images[index])
        # gt = cv2.imread(self.gt_images[index])
        # input_resized, gt = data_aug(input_src, gt)
        #
        # input_resized = input_resized.astype(dtype=np.float32)
        # input_resized = input_resized / 255.0
        # input_resized = np.transpose(input_resized, [2, 0, 1])
        # gt = gt.astype(dtype=np.float32)
        # gt = gt / 255.0
        # gt = np.transpose(gt, [2, 0, 1])
        #
        # '''load mask'''
        # mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        # mask = np.expand_dims(mask, axis=-1)
        # mask = mask.astype(dtype=np.float32)
        # mask = mask / 255.0
        # mask = np.transpose(mask, [2, 0, 1])
        # mask_tensor = torch.tensor(mask)
        #
        # '''load flow'''
        # flow = np.load(self.flows[index])
        # flow = flow.astype(dtype=np.float32)
        # flow_tensor = torch.tensor(flow)
        #
        #
        # input_tensor = torch.tensor(input_resized)
        # gt_tensor = torch.tensor(gt)
        # task_id_tensor = torch.tensor(task_id, dtype=torch.int64)


        # face_mask_path = self.input_images[index].replace('/input/', '/mask_face/')
        # facemask = cv2.imread(face_mask_path, 0)
        # facemask = facemask.astype(dtype=np.float32)
        # facemask = (facemask / 255.0)
        # facemask = np.expand_dims(facemask, axis=-1)
        # facemask = np.transpose(facemask, [2, 0, 1])
        # face_mask = torch.tensor(facemask)
        # mask_sum = torch.sum(face_mask)
        # weight = self.width * self.height / mask_sum - 1
        # weight = torch.max(weight / 3, torch.ones(1))
        # face_weight = weight.unsqueeze(-1).unsqueeze(-1)

        input_src = cv2.imread(self.input_images[index])
        gt = cv2.imread(self.gt_images[index])
        input_resized, gt = data_aug(input_src, gt)

        input_resized = input_resized.astype(dtype=np.float32)
        input_resized = input_resized / 255.0
        input_resized = np.transpose(input_resized, [2, 0, 1])
        gt = gt.astype(dtype=np.float32)
        gt = gt / 255.0
        gt = np.transpose(gt, [2, 0, 1])

        '''load mask'''
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(dtype=np.float32)
        mask = mask / 255.0
        mask = np.transpose(mask, [2, 0, 1])
        mask_tensor = torch.tensor(mask)

        '''load flow'''
        flow = np.load(self.flows[index])
        flow = flow.astype(dtype=np.float32)
        flow_tensor = torch.tensor(flow)

        input_tensor = torch.tensor(input_resized)
        gt_tensor = torch.tensor(gt)
        task_id_tensor = torch.tensor(task_id, dtype=torch.int64)

        flow_tensor = F.pad(flow_tensor, pad=padding, mode='constant', value=0)
        gt_tensor = F.pad(gt_tensor, pad=padding, mode='constant', value=0)
        input_tensor = F.pad(input_tensor, pad=padding, mode='constant', value=0)
        mask_tensor = F.pad(mask_tensor, pad=padding, mode='constant', value=0)

        face_mask = torch.zeros_like(mask_tensor)
        face_weight = torch.mean(face_mask).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return (input_tensor, gt_tensor, mask_tensor, task_id_tensor, flow_tensor, face_mask, face_weight)

    def __len__(self):
        return len(self.input_images)


class Fine_Tuning_TestDataSet(Dataset):
    def __init__(self, data_path, task_id, unifide_size):
        self.width = unifide_size[0]
        self.height = unifide_size[1]
        self.test_path = data_path
        self.datas = OrderedDict()
        self.task_id = task_id


        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'gt' or data_name == 'mask':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.*'))
                self.datas[data_name]['image'].sort()

    def __getitem__(self, index):

        '''load images'''
        padding = (128, 128, 64, 64)  # 左右各128，上下各64
        if (self.task_id == 3) or (self.task_id == 4):
            input = cv2.imread(self.datas['input']['image'][index])
            input = cv2.resize(input, (self.width, self.height))
            input = input.astype(dtype=np.float32)
            input = input / 255.0
            input = np.transpose(input, [2, 0, 1])

            gt = cv2.imread(self.datas['gt']['image'][index])
            gt = cv2.resize(gt, (self.width, self.height))
            gt = gt.astype(dtype=np.float32)
            gt = gt / 255.0
            gt = np.transpose(gt, [2, 0, 1])

            mask = cv2.imread(self.datas['mask']['image'][index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.width, self.height))
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(dtype=np.float32)
            mask = mask / 255.0
            mask = np.transpose(mask, [2, 0, 1])

            input_tensor = torch.tensor(input)
            gt_tensor = torch.tensor(gt)
            mask_tensor = torch.tensor(mask)
            task_id_tensor = torch.tensor(self.task_id, dtype=torch.int64)
        elif (self.task_id == 0) or (self.task_id == 1) or (self.task_id == 2):
            input = cv2.imread(self.datas['input']['image'][index])
            input = cv2.resize(input, (self.width, self.height))
            input = input.astype(dtype=np.float32)
            input = input / 255.0
            input = np.transpose(input, [2, 0, 1])

            gt = cv2.imread(self.datas['gt']['image'][index])
            gt = cv2.resize(gt, (self.width, self.height))
            gt = gt.astype(dtype=np.float32)
            gt = gt / 255.0
            gt = np.transpose(gt, [2, 0, 1])

            mask = cv2.imread(self.datas['mask']['image'][index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.width, self.height))
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(dtype=np.float32)
            mask = mask / 255.0
            mask = np.transpose(mask, [2, 0, 1])

            input_tensor = torch.tensor(input)
            gt_tensor = torch.tensor(gt)
            mask_tensor = torch.tensor(mask)
            task_id_tensor = torch.tensor(self.task_id, dtype=torch.int64)

            gt_tensor = F.pad(gt_tensor, pad=padding, mode='constant', value=0)
            input_tensor = F.pad(input_tensor, pad=padding, mode='constant', value=0)
            mask_tensor = F.pad(mask_tensor, pad=padding, mode='constant', value=0)

        file_name = os.path.basename(self.datas['input']['image'][index])
        file_name, _ = os.path.splitext(file_name)




        return (input_tensor, gt_tensor, mask_tensor, task_id_tensor)

    def __len__(self):
        return len(self.datas['input']['image'])





class ftfourTestDataSet(Dataset):
    def __init__(self, data_path, task_id, unifide_size):
        self.width = unifide_size[0]
        self.height = unifide_size[1]
        self.test_path = data_path
        self.datas = OrderedDict()
        self.task_id = task_id


        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'gt' or data_name == 'mask':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.*'))
                self.datas[data_name]['image'].sort()

    def __getitem__(self, index):

        '''load images'''
        padding = (128, 128, 64, 64)  # 左右各128，上下各64

        input = cv2.imread(self.datas['input']['image'][index])
        input = cv2.resize(input, (self.width, self.height))
        input = input.astype(dtype=np.float32)
        input = input / 255.0
        input = np.transpose(input, [2, 0, 1])

        gt = cv2.imread(self.datas['gt']['image'][index])
        gt = cv2.resize(gt, (self.width, self.height))
        gt = gt.astype(dtype=np.float32)
        gt = gt / 255.0
        gt = np.transpose(gt, [2, 0, 1])

        mask = cv2.imread(self.datas['mask']['image'][index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.width, self.height))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(dtype=np.float32)
        mask = mask / 255.0
        mask = np.transpose(mask, [2, 0, 1])

        input_tensor = torch.tensor(input)
        gt_tensor = torch.tensor(gt)
        mask_tensor = torch.tensor(mask)
        task_id_tensor = torch.tensor(self.task_id, dtype=torch.int64)

        gt_tensor = F.pad(gt_tensor, pad=padding, mode='constant', value=0)
        input_tensor = F.pad(input_tensor, pad=padding, mode='constant', value=0)
        mask_tensor = F.pad(mask_tensor, pad=padding, mode='constant', value=0)

        file_name = os.path.basename(self.datas['input']['image'][index])
        file_name, _ = os.path.splitext(file_name)


        return (input_tensor, gt_tensor, mask_tensor, task_id_tensor)

    def __len__(self):
        return len(self.datas['input']['image'])
