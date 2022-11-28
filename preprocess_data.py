import glob
import argparse
import numpy as np
import os
import torch
import random
from tqdm import tqdm
import json
import functools
from utilities import HelperFunctions
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import pickle

torch.multiprocessing.set_sharing_strategy('file_system')


class CropMAMLDataLoader_v3(Dataset):
    def __init__(self, crop, split, num_samples_per_support, num_samples_per_query, num_tasks, options=None, data_dir=None):
        '''
        options==special means use add train support, train query, test support as training set only for baselines
        '''
        if not data_dir:
            self.data_dir = f"./meta/data/block_sample/{crop}"
            print(f"using dataset data/block_sample")
        else:
            self.data_dir = os.path.join(data_dir, crop)
            print(f"using dataset {data_dir}")
        self.num_samples_per_support = num_samples_per_support
        self.num_samples_per_query = num_samples_per_query
        self.crop = crop
        self.split = split
        self.fips_ids = []
        for file_name in os.listdir(os.path.join(self.data_dir, self.split)):
            self.fips_ids.append(file_name.split(".")[0])

        self.options = options
        if options == "special_baseline":
            for file_name in os.listdir(os.path.join(self.data_dir, "test")):
                self.fips_ids.append(file_name.split(".")[0])

        if num_tasks==0:
            self.task_num = len(self.fips_ids)
        elif num_tasks <= len(self.fips_ids):
            self.task_num = num_tasks
        else:
            self.task_num = len(self.fips_ids)


    def __len__(self):
        return self.task_num

    def __getitem__(self, i):
        if self.split == "train" and self.options == "special_baseline":
            fips_id = self.fips_ids[i]
            if os.path.exists(os.path.join(self.data_dir, self.split, f"{fips_id}.pkl")):
                with open(os.path.join(self.data_dir, self.split, f"{fips_id}.pkl"), "rb") as f:
                    data = pickle.load(f)
                support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = self.get_samples(data)
            else:
                with open(os.path.join(self.data_dir, "test", f"{fips_id}.pkl"), "rb") as f:
                    data = pickle.load(f)
                support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = self.get_samples(data)
                query_x, query_y, query_fips, query_dp, query_year = torch.zeros_like(query_x), torch.zeros_like(query_y), torch.zeros_like(query_fips), torch.zeros_like(query_dp), torch.zeros_like(query_year)

            return support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year

        else:
            fips_id = self.fips_ids[i]
            with open(os.path.join(self.data_dir, self.split, f"{fips_id}.pkl"), "rb") as f:
                data = pickle.load(f)

            support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = self.get_samples(data)

            return support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year

    def get_samples(self, data, add_year_feature=False):
        num_samples_per_support = self.num_samples_per_support
        num_samples_per_query = self.num_samples_per_query

        support = data["support"]
        query = data["query"]

        support_x = support["x"]
        support_y = support["y"]
        support_fips = torch.Tensor(support["fips"])
        support_dp = torch.Tensor(support["dp"])
        support_year = torch.Tensor(support["year"])

        random.seed(22)
        if num_samples_per_support > len(support_dp):
            support_index = random.sample(range(0, len(support_dp)), len(support_dp))
            offset = num_samples_per_support - len(support_dp)
            while offset > 0:
                idx = offset % len(support_dp)
                support_index.append(support_index[idx])
                offset -= 1
            print(support_index, "support")
        else:
            support_index = random.sample(range(0, len(support_dp)), num_samples_per_support)



        support_x_sample = support_x[support_index, :]
        support_y_sample = support_y[support_index, :]
        support_fips_sample = support_fips[support_index]
        support_dp_sample = support_dp[support_index]
        support_year_sample = support_year[support_index]

        query_x = query["x"]
        query_y = query["y"]
        query_fips = torch.Tensor(query["fips"])
        query_dp = torch.Tensor(query["dp"])
        query_year = torch.Tensor(query["year"])

        random.seed(22)
        if num_samples_per_query > len(query_dp):
            query_index = random.sample(range(0, len(query_dp)), len(query_dp))
            offset = num_samples_per_query - len(query_dp)
            while offset > 0:
                idx = offset % len(query_dp)
                query_index.append(query_index[idx])
                offset -= 1
            print(query_index, "query")
        else:
            query_index = random.sample(range(0, len(query_dp)), num_samples_per_query)


        query_x_sample = query_x[query_index, :]
        query_y_sample = query_y[query_index, :]
        query_fips_sample = query_fips[query_index]
        query_dp_sample = query_dp[query_index]
        query_year_sample = query_year[query_index]

        return support_x_sample, support_y_sample, support_fips_sample, support_dp_sample, support_year_sample, query_x_sample, query_y_sample, query_fips_sample, query_dp_sample, query_year_sample


def collate_fn(batch, dataset):
    original_batch_len = len(batch)
    batch = list(filter(lambda x: x[0] is not None, batch))
    filtered_batch_len = len(batch)
    diff = original_batch_len - filtered_batch_len
    # only if filter out all original samples, it run re-sampling
    if diff == original_batch_len:
        batch.extend([dataset[random.randint(0, len(dataset))] for _ in range(diff)])
        return collate_fn(batch, dataset)
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataloader():
    train_dataset = CropMAMLDataLoader_v3(args.crop, "train", num_samples_per_support=args.num_per_support,
                                          num_samples_per_query=args.num_per_query, num_tasks=args.num_tasks,
                                          data_dir=args.data_dir)
    test_dataset = CropMAMLDataLoader_v3(args.crop, "test", num_samples_per_support=args.num_per_support,
                                         num_samples_per_query=args.num_per_query, num_tasks=args.num_tasks,
                                         data_dir=args.data_dir)

    collate_fn_train = functools.partial(collate_fn, dataset=train_dataset)
    collate_fn_test = functools.partial(collate_fn, dataset=test_dataset)

    dataloader_train = DataLoader(train_dataset, batch_size=args.task_per_batch, shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn_train)
    dataloader_test = DataLoader(test_dataset, batch_size=args.task_per_batch, shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn_test)

    return dataloader_train, dataloader_test

def get_device():
    gpu_ids = [int(i) for i in args.gpus.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpu_ids[0]}"
    device = torch.device("cuda")
    print(f"use device {gpu_ids} gpus")
    return device, gpu_ids


def get_argparser():
    parser = argparse.ArgumentParser(description='Meta Learning')
    parser.add_argument('--data-dir', default='./data/block_sample/', help='path to data')
    parser.add_argument('--save-dir', default='./data/fine_sample/', help='path save used dataset on disk')
    parser.add_argument('--num-workers', default=8, type=int, help='number of workers used in dataloader')
    parser.add_argument('--crop', default="soybean", choices=["corn", "soybean"], help='crop category')
    parser.add_argument('--gpus', default='3', type=str, help='specified gpus')
    parser.add_argument('--seed', default=20, type=int, help="random seed number")
    parser.add_argument('--num-per-support', default=25, type=int, help='number of samples per support set')
    parser.add_argument('--num-per-query', default=75, type=int, help='number of samples per query set')
    parser.add_argument('--task-per-batch', default=1, type=int, help='number of tasks per batch')
    parser.add_argument('--num-tasks', default=0, type=int, help='number of tasks to set')

    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    device, gpu_ids = get_device()
    dataloader_train, dataloader_test = get_dataloader()
    res_dict = {"train":{},"test":{}}
    for iter, batch in tqdm(enumerate(dataloader_train)):
        batch_tasks = batch[0].shape[0]
        batch_pred = []
        batch_gold = []
        for i in range(batch_tasks):
            support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i], batch[5][i], batch[6][i], \
                batch[7][i], batch[8][i], batch[9][i]
            res_dict["train"][int(support_fips[0])] = batch

    for iter, batch in tqdm(enumerate(dataloader_test)):
        batch_tasks = batch[0].shape[0]
        batch_pred = []
        batch_gold = []
        for i in range(batch_tasks):
            support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i], batch[5][i], batch[6][i], \
                batch[7][i], batch[8][i], batch[9][i]
            res_dict["test"][int(support_fips[0])] = batch


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, f"{args.crop}_block_sample.pkl"), "wb") as f:
        pickle.dump(res_dict, f)

