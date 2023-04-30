import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from collections import namedtuple


Batch = namedtuple('Batch', 'Observations Actions Class')


class PlanningDataset(Dataset):
    """
    load video and action features from dataset
    """

    def __init__(self,
                 root,
                 args=None,
                 is_val=False,
                 model=None,
                 ):
        self.is_val = is_val
        self.data_root = root
        self.args = args
        self.max_traj_len = args.horizon
        self.vid_names = None
        self.frame_cnts = None
        self.images = None
        self.last_vid = ''

        if args.dataset == 'crosstask':
            if is_val:
                cross_task_data_name = args.json_path_val
                    # "/data1/wanghanlin/diffusion_planning/jsons_crosstask105/sliding_window_cross_task_data_{}_{}_new_task_id_73_with_event_class.json".format(is_val, self.max_traj_len)
            else:
                cross_task_data_name = args.json_path_train
                    # "/data1/wanghanlin/diffusion_planning/jsons_crosstask105/sliding_window_cross_task_data_{}_{}_new_task_id_73.json".format(
                    # is_val, self.max_traj_len)

            if os.path.exists(cross_task_data_name):
                with open(cross_task_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(cross_task_data_name))
            else:
                assert 0
        elif args.dataset == 'coin':
            if is_val:
                coin_data_name = args.json_path_val
                    # "/data1/wanghanlin/diffusion_planning/jsons_coin/sliding_window_cross_task_data_{}_{}_new_task_id_73_with_event_class.json".format(
                    # is_val, self.max_traj_len)
            else:
                coin_data_name = args.json_path_train
                    # "/data1/wanghanlin/diffusion_planning/jsons_coin/sliding_window_cross_task_data_{}_{}_new_task_id_73.json".format(
                    # is_val, self.max_traj_len)
            if os.path.exists(coin_data_name):
                with open(coin_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(coin_data_name))
            else:
                assert 0
        elif args.dataset == 'NIV':
            if is_val:
                niv_data_name = args.json_path_val
                    # "/data1/wanghanlin/diffusion_planning/jsons_niv/sliding_window_cross_task_data_{}_{}_new_task_id_73_with_event_class.json".format(
                    # is_val, self.max_traj_len)
            else:
                niv_data_name = args.json_path_train
                    # "/data1/wanghanlin/diffusion_planning/jsons_niv/sliding_window_cross_task_data_{}_{}_new_task_id_73.json".format(
                    # is_val, self.max_traj_len)
            if os.path.exists(niv_data_name):
                with open(niv_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(niv_data_name))
            else:
                assert 0
        else:
            raise NotImplementedError(
                'Dataset {} is not implemented'.format(args.dataset))

        self.model = model
        self.prepare_data()
        self.M = 3

    def prepare_data(self):
        vid_names = []
        frame_cnts = []
        for listdata in self.json_data:
            vid_names.append(listdata['id'])
            frame_cnts.append(listdata['instruction_len'])
        self.vid_names = vid_names
        self.frame_cnts = frame_cnts

    def curate_dataset(self, images, legal_range, M=2):
        images_list = []
        labels_onehot_list = []
        idx_list = []
        for start_idx, end_idx, action_label in legal_range:
            idx = start_idx
            idx_list.append(idx)
            image_start_idx = max(0, idx)
            if image_start_idx + M <= len(images):
                image_start = images[image_start_idx: image_start_idx + M]
            else:
                image_start = images[len(images) - M: len(images)]
            image_start_cat = image_start[0]
            for w in range(len(image_start) - 1):
                image_start_cat = np.concatenate((image_start_cat, image_start[w + 1]), axis=0)
            images_list.append(image_start_cat)
            labels_onehot_list.append(action_label)

        end_idx = max(2, end_idx)
        image_end = images[end_idx - 2:end_idx + M - 2]
        image_end_cat = image_end[0]
        for w in range(len(image_end) - 1):
            image_end_cat = np.concatenate((image_end_cat, image_end[w + 1]), axis=0)
        images_list.append(image_end_cat)
        return images_list, labels_onehot_list, idx_list

    def sample_single(self, index):
        folder_id = self.vid_names[index]

        if self.is_val:
            event_class = folder_id['event_class']
        else:
            task_class = folder_id['task_id']

        if self.args.dataset == 'crosstask':
            if folder_id['vid'] != self.last_vid:
                images_ = np.load(folder_id['feature'], allow_pickle=True)
                self.images = images_['frames_features']
                self.last_vid = folder_id['vid']
        else:
            images_ = np.load(folder_id['feature'], allow_pickle=True)
            self.images = images_['frames_features']
        images, labels_matrix, idx_list = self.curate_dataset(
            self.images, folder_id['legal_range'], M=self.M)
        frames = torch.tensor(np.array(images))
        labels_tensor = torch.tensor(labels_matrix, dtype=torch.long)

        if self.is_val:
            event_class = torch.tensor(event_class, dtype=torch.long)
            return frames, labels_tensor, event_class
        else:
            task_class = torch.tensor(task_class, dtype=torch.long)
            return frames, labels_tensor, task_class

    def __getitem__(self, index):
        if self.is_val:
            frames, labels, event_class = self.sample_single(index)
        else:
            frames, labels, task = self.sample_single(index)
        if self.is_val:
            batch = Batch(frames, labels, event_class)
        else:
            batch = Batch(frames, labels, task)
        return batch

    def __len__(self):
        return len(self.json_data)