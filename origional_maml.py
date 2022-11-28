import argparse
import shutil
import time
import learn2learn as l2l
import pandas as pd
from tensorboard_logger import configure, log_value
from torch import nn, optim
import numpy as np
import os
import torch
import json
import pickle
from models import CornYieldModel
from utilities import AverageMeter, ProgressMeter, HelperFunctions, R2Loss


def get_batch_data(data, fips_list, batch_size):
    batch_data = []
    one_batch = []
    for i, fips in enumerate(fips_list):
        if i > 0 and i % batch_size == 0:
            tmp_batch = []
            for k in range(len(one_batch[0])):
                tmp_arr = np.empty([len(one_batch), len(one_batch[0])], dtype=object)
                for m in range(len(one_batch)):
                    for n in range(len(one_batch[0])):
                        tmp_arr[m, n] = one_batch[m][n]
                tmp_batch.append(torch.cat(list(tmp_arr[:, k]), dim=0))
            batch_data.append(tmp_batch)
            one_batch = []
        task = data[fips]
        one_batch.append(task)
    if len(one_batch) > 0:
        tmp_batch = []
        for k in range(len(one_batch[0])):
            tmp_arr = np.empty([len(one_batch), len(one_batch[0])], dtype=object)
            for m in range(len(one_batch)):
                for n in range(len(one_batch[0])):
                    tmp_arr[m, n] = one_batch[m][n]
            tmp_batch.append(torch.cat(list(tmp_arr[:, k]), dim=0))
        batch_data.append(tmp_batch)
    return batch_data


def get_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def main():
    exp_postfix = f"support_size_{args.num_per_support}_query_size_{args.num_per_query}"

    with open(os.path.join(args.exp_dir, 'configs.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    data = get_data(f"./data/fine_sample/{args.crop}_block_sample.pkl")
    train_data = data["train"]
    train_fips = list(train_data.keys())
    test_data = data["test"]
    test_fips = list(test_data.keys())

    dataloader_test = get_batch_data(test_data, test_fips, args.task_per_batch)
    dataloader_train = get_batch_data(train_data, train_fips, args.task_per_batch)

    checkpoint = torch.load(args.pretrained_model)
    model = CornYieldModel(num_features=19, hidden_size=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    configure(args.exp_dir)

    maml = l2l.algorithms.MAML(model, lr=args.adapt_lr, first_order=False, allow_unused=True, allow_nograd=False)
    # maml = MAML(model, lr=args.adapt_lr, first_order=False, allow_unused=True, allow_nograd=False)
    optimizer = optim.Adam(maml.parameters(), args.meta_lr)
    lossfn = nn.MSELoss(reduction='mean')
    compute_r2 = R2Loss()

    best_train_r2 = 0
    for epoch in range(args.num_epochs):
        for phase in ["train", "test"]:
            if phase == "train":
                pred_arr = []
                gold_arr = []
                fips_id_arr = []
                year_id_arr = []

                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses = AverageMeter('Loss', ':.4e')
                r2 = AverageMeter('R2', ':.4e')
                progress = ProgressMeter(len(dataloader_train), [batch_time, data_time, losses, r2],
                                         prefix="Train Epoch: [{}]".format(epoch))

                end = time.time()
                for iter, batch in enumerate(dataloader_train):
                    data_time.update(time.time() - end)
                    meta_train_loss = 0.0

                    # for each task in the batch
                    batch_tasks = batch[0].shape[0]
                    batch_pred = []
                    batch_gold = []
                    for i in range(batch_tasks):
                        learner = maml.clone()
                        if len(gpu_ids) > 1:
                            learner.module = torch.nn.DataParallel(learner.module, device_ids=gpu_ids)

                        support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                            batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i], batch[5][i], batch[6][i], \
                            batch[7][i], batch[8][i], batch[9][i]

                        for _ in range(args.adapt_steps):
                            support_preds = learner(support_x.to(device)).squeeze(2)
                            support_loss = lossfn(support_preds, support_y.to(device))
                            learner.adapt(support_loss, allow_unused=True, allow_nograd=False)

                        query_preds = learner(query_x.to(device)).squeeze(2)
                        query_loss = lossfn(query_preds, query_y.to(device))
                        meta_train_loss += query_loss

                        # print('Train Epoch',epoch,'Batch:', iter, "Task", i, 'query Loss', query_loss.item())

                        pred_arr.append(query_preds)
                        gold_arr.append(query_y)
                        fips_id_arr.append(query_fips)
                        year_id_arr.append(query_year)

                        batch_pred.append(query_preds)
                        batch_gold.append(query_y)

                        optimizer.zero_grad()
                        query_loss.backward()
                        optimizer.step()

                    meta_train_loss = meta_train_loss / batch_tasks

                    batch_time.update(time.time() - end)
                    losses.update(meta_train_loss.item(), batch_tasks)
                    batch_pred = torch.cat(batch_pred)
                    batch_gold = torch.cat(batch_gold)
                    local_r2 = compute_r2(batch_pred, batch_gold.to(device)).detach().cpu().numpy()
                    r2.update(local_r2, batch_tasks)

                    if iter % 1 == 0:
                        # print('Train Epoch', epoch, 'Batch', iter, 'Meta Train Loss', meta_train_loss.item())
                        progress.display(iter)
                        step = iter + len(dataloader_train) * epoch
                        log_value('train/epoch', epoch, step)
                        log_value('train/loss', progress.meters[2].avg, step)

                helper = HelperFunctions()
                compute_r2 = R2Loss()

                pred_arr = torch.cat(pred_arr).cpu().squeeze()
                gold_arr = torch.cat(gold_arr).cpu().squeeze()
                fips_id_arr = torch.cat(fips_id_arr).cpu().squeeze()
                year_id_arr = torch.cat(year_id_arr).cpu().squeeze()

                pred_res = helper.Z_norm_reverse(pred_arr, helper.scalar[0])
                gold_res = helper.Z_norm_reverse(gold_arr, helper.scalar[0])

                df = pd.DataFrame(
                    {"fips_id": fips_id_arr.tolist(), "year": year_id_arr.tolist(), "pred": pred_res.tolist(),
                     "gold": gold_res.tolist()})
                df = df.groupby(['fips_id', 'year'], as_index=False).mean()
                df = df.sort_values(["fips_id", "year"], ascending=(True, True))
                df.to_csv(os.path.join(args.exp_dir, f"result_train_epoch_{epoch}_{exp_postfix}.csv"))

                # print(df)

                pred_res, gold_res = np.array(df["pred"]), np.array(df["gold"])
                pred_res = torch.from_numpy(pred_res).to(device)
                gold_res = torch.from_numpy(gold_res).to(device)

                helper.plot(pred_res, gold_res, args.exp_dir, f"plot_train_epoch_{epoch}_{exp_postfix}")

                R2 = compute_r2(pred_res, gold_res).detach().cpu().numpy()
                # print("normalized R2", R2)

                if epoch % 1 == 0:
                    log_value('train/r2', R2, epoch)

                if R2 > best_train_r2:
                    is_best = True
                    best_train_r2 = R2
                else:
                    is_best = False

                ckpt_name = 'model_epoch_' + str(epoch) + '.pkl'
                file_name = os.path.join(args.exp_dir, ckpt_name)
                save_checkpoint(args, {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'r2': R2,
                    'optimizer': optimizer.state_dict(),
                }, is_best, file_name)

            if phase == "test":
                helper = HelperFunctions()
                compute_r2 = R2Loss()

                pred_arr = []
                gold_arr = []
                fips_id_arr = []
                year_id_arr = []

                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses = AverageMeter('Loss', ':.4e')
                r2 = AverageMeter('R2', ':.4e')
                progress = ProgressMeter(len(dataloader_test), [batch_time, data_time, losses, r2],
                                         prefix="Test Epoch: [{}]".format(epoch))

                end = time.time()
                for iter, batch in enumerate(dataloader_test):
                    data_time.update(time.time() - end)
                    meta_test_loss = 0.0

                    batch_tasks = batch[0].shape[0]
                    batch_pred = []
                    batch_gold = []
                    for i in range(batch_tasks):
                        # set first_order=True in test in order to prevent memory leaks in gpus
                        learner = maml.clone()
                        if len(gpu_ids) > 1:
                            learner.module = torch.nn.DataParallel(learner.module, device_ids=gpu_ids)

                        support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                            batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i], batch[5][i], batch[6][i], \
                            batch[7][i], batch[8][i], batch[9][i]

                        for _ in range(args.adapt_steps + args.adapt_steps_extra):
                            # with torch.no_grad():
                            support_preds = learner(support_x.to(device)).squeeze(2)
                            # support_preds.requires_grad=True
                            support_loss = lossfn(support_preds, support_y.to(device))
                            # evaluation ==True has a bug here
                            learner.adapt(support_loss, allow_unused=True, allow_nograd=True)

                        with torch.no_grad():
                            query_preds = learner(query_x.to(device)).squeeze(2)
                        # query_preds.requires_grad=True
                        query_loss = lossfn(query_preds, query_y.to(device))
                        meta_test_loss += query_loss

                        pred_arr.append(query_preds)
                        gold_arr.append(query_y)
                        fips_id_arr.append(query_fips)
                        year_id_arr.append(query_year)

                        batch_pred.append(query_preds)
                        batch_gold.append(query_y)

                        # print('Test Epoch',epoch,'Batch:', iter, "Task", i, 'query Loss', query_loss.item())

                    meta_test_loss = meta_test_loss / batch_tasks

                    batch_time.update(time.time() - end)
                    losses.update(meta_test_loss.item(), batch_tasks)
                    batch_pred = torch.cat(batch_pred)
                    batch_gold = torch.cat(batch_gold)
                    local_r2 = compute_r2(batch_pred, batch_gold.to(device)).detach().cpu().numpy()
                    r2.update(local_r2, batch_tasks)

                    if iter % 1 == 0:
                        # print('Test Epoch', epoch, 'Batch', iter, 'Meta Test Loss', meta_test_loss.item())
                        progress.display(iter)
                        step = iter + len(dataloader_test) * epoch
                        log_value('test/epoch', epoch, step)
                        log_value('test/loss', progress.meters[2].avg, step)

                pred_arr = torch.cat(pred_arr).cpu().squeeze()
                gold_arr = torch.cat(gold_arr).cpu().squeeze()
                fips_id_arr = torch.cat(fips_id_arr).cpu().squeeze()
                year_id_arr = torch.cat(year_id_arr).cpu().squeeze()

                pred_res = helper.Z_norm_reverse(pred_arr, helper.scalar[0])
                gold_res = helper.Z_norm_reverse(gold_arr, helper.scalar[0])

                df = pd.DataFrame(
                    {"fips_id": fips_id_arr.tolist(), "year": year_id_arr.tolist(), "pred": pred_res.tolist(),
                     "gold": gold_res.tolist()})
                df = df.groupby(['fips_id', 'year'], as_index=False).mean()
                df = df.sort_values(["fips_id", "year"], ascending=(True, True))
                df.to_csv(os.path.join(args.exp_dir, f"result_test_epoch_{epoch}_{exp_postfix}.csv"))
                # print(df)

                pred_res, gold_res = np.array(df["pred"]), np.array(df["gold"])
                pred_res = torch.from_numpy(pred_res).to(device)
                gold_res = torch.from_numpy(gold_res).to(device)

                helper.plot(pred_res, gold_res, args.exp_dir, f"plot_test_epoch_{epoch}_{exp_postfix}")

                R2 = compute_r2(pred_res, gold_res).detach().cpu().numpy()
                # print("normalized R2", R2)

                if epoch % 1 == 0:
                    log_value('test/r2', R2, epoch)


def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.exp_dir, 'model_best.pkl'))


def get_device():
    gpu_ids = [int(i) for i in args.gpus.split(",")]
    device = torch.device("cuda")
    print(f"use device {gpu_ids} gpus")
    return device, gpu_ids


def get_argparser():
    parser = argparse.ArgumentParser(description='Meta Learning')
    parser.add_argument('--data-dir', default='./data/block_sample_v6/', help='path to data')
    parser.add_argument('--pretrained-model', default='./checkpoints/best_model_synthetic_split_by_space_v2.pkl',
                        help='path to pretrained model')
    parser.add_argument('--exp-dir', default='./experiments/debug', help='path save experimental results')
    parser.add_argument('--num-workers', default=8, type=int, help='number of workers used in dataloader')
    parser.add_argument('--crop', default="corn", choices=["corn", "soybean"], help='crop category')
    parser.add_argument('--num-epochs', default=20, type=int, help='number of running epochs')
    parser.add_argument('--gpus', default='1', type=str, help='specified gpus')
    parser.add_argument('--seed', default=20, type=int, help="random seed number")
    parser.add_argument('--num-per-support', default=25, type=int, help='number of samples per support set')
    parser.add_argument('--num-per-query', default=75, type=int, help='number of samples per query set')
    parser.add_argument('--task-per-batch', default=32, type=int, help='number of tasks per batch')
    parser.add_argument('--num-tasks', default=64, type=int, help='number of tasks to set')
    parser.add_argument('--adapt-lr', default=0.001, type=float, help='adaptive learning rate')
    parser.add_argument('--meta-lr', default=0.001, type=float, help='meta learning rate')
    parser.add_argument('--adapt-steps', default=1, type=int, help='adaptive steps')
    parser.add_argument('--adapt-steps-extra', default=1, type=int, help='addictive adaptive steps in test finetune')

    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    device, gpu_ids = get_device()
    main()
