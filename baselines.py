import argparse
import copy
import shutil
import time
import pandas as pd
from tensorboard_logger import configure, log_value
from torch import nn, optim
import numpy as np
import os
import torch
import json
import pickle
from models import CornYieldModel
from utilities import AverageMeter, ProgressMeter, R2Loss, HelperFunctions


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

    optimizer = optim.Adam(model.parameters(), args.meta_lr)
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
                model.train()

                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses = AverageMeter('Loss', ':.4e')
                r2 = AverageMeter('R2', ':.4e')
                progress = ProgressMeter(len(dataloader_train), [batch_time, data_time, losses, r2],
                                         prefix="Train Epoch: [{}]".format(epoch))

                train_loss = 0.0
                end = time.time()
                for iter, batch in enumerate(dataloader_train):
                    data_time.update(time.time() - end)

                    support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                        batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[
                            9]
                    if args.baseline_mode == 0:
                        # use train support set
                        X = support_x.reshape(-1, support_x.shape[-2], support_x.shape[-1])
                        y = support_y.reshape(-1, support_y.shape[-1])
                        fips = support_fips.reshape(-1, )
                        dp = support_dp.reshape(-1, )
                        year = support_year.reshape(-1, )
                    else:  # args.baseline_mode in [1,2,3]:
                        # concatenate train support and query
                        X = torch.cat([support_x, query_x], dim=1).reshape(-1, support_x.shape[-2], support_x.shape[-1])
                        y = torch.cat([support_y, query_y], dim=1).reshape(-1, support_y.shape[-1])
                        fips = torch.cat([support_fips, query_fips], dim=1).reshape(-1, )
                        dp = torch.cat([support_dp, query_dp], dim=1).reshape(-1, )
                        year = torch.cat([support_year, query_year], dim=1).reshape(-1, )

                    # run model
                    y_pred = model(X.to(device)).squeeze(2)
                    loss = lossfn(y_pred, y.to(device))

                    # save prediction
                    pred_arr.append(y_pred)
                    gold_arr.append(y)
                    fips_id_arr.append(fips)
                    year_id_arr.append(year)

                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    batch_time.update(time.time() - end)
                    losses.update(loss.item(), X.shape[0])
                    local_r2 = compute_r2(y_pred, y.to(device)).detach().cpu().numpy()
                    r2.update(local_r2, X.shape[0])

                    if iter % 1 == 0:
                        # print('Train Epoch', epoch, 'Batch', iter, 'Train Loss', train_loss)
                        progress.display(iter + 1)
                        step = iter + len(dataloader_train) * epoch
                        log_value('train/epoch', epoch, step)
                        log_value('train/loss', progress.meters[2].avg, step)

                if args.baseline_mode == 2:
                    for iter, batch in enumerate(dataloader_test):
                        data_time.update(time.time() - end)

                        support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                            batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], \
                            batch[
                                9]
                        # use test support set
                        X = support_x.reshape(-1, support_x.shape[-2], support_x.shape[-1])
                        y = support_y.reshape(-1, support_y.shape[-1])
                        fips = support_fips.reshape(-1, )
                        dp = support_dp.reshape(-1, )
                        year = support_year.reshape(-1, )

                        # run model
                        y_pred = model(X.to(device)).squeeze(2)
                        loss = lossfn(y_pred, y.to(device))

                        # save prediction
                        pred_arr.append(y_pred)
                        gold_arr.append(y)
                        fips_id_arr.append(fips)
                        year_id_arr.append(year)

                        # optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()

                        batch_time.update(time.time() - end)
                        losses.update(loss.item(), X.shape[0])
                        local_r2 = compute_r2(y_pred, y.to(device)).detach().cpu().numpy()
                        r2.update(local_r2, X.shape[0])

                        if iter % 1 == 0:
                            # print('Train Epoch', epoch, 'Batch', iter, 'Train Loss', train_loss)
                            progress.display(iter + 1)
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
                model_test = copy.deepcopy(model)

                if args.baseline_mode == 3:  # finetune on the test query
                    for iter, batch in enumerate(dataloader_test):
                        support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                            batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], \
                            batch[
                                9]
                        # use test support set
                        X = support_x.reshape(-1, support_x.shape[-2], support_x.shape[-1])
                        y = support_y.reshape(-1, support_y.shape[-1])
                        fips = support_fips.reshape(-1, )
                        dp = support_dp.reshape(-1, )
                        year = support_year.reshape(-1, )

                        # run model
                        y_pred = model_test(X.to(device)).squeeze(2)
                        loss = lossfn(y_pred, y.to(device))

                        # optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model_test.eval()

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
                test_loss = 0
                for iter, batch in enumerate(dataloader_test):
                    data_time.update(time.time() - end)

                    support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                        batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[
                            9]

                    # use query data as test data
                    X = query_x.reshape(-1, query_x.shape[-2], query_x.shape[-1])
                    y = query_y.reshape(-1, query_y.shape[-1])
                    fips = query_fips.reshape(-1, )
                    dp = query_dp.reshape(-1, )
                    year = query_year.reshape(-1, )

                    # run model
                    with torch.no_grad():
                        y_pred = model_test(X.to(device)).squeeze(2)
                    loss = lossfn(y_pred, y.to(device))

                    # save prediction
                    pred_arr.append(y_pred)
                    gold_arr.append(y)
                    fips_id_arr.append(fips)
                    year_id_arr.append(year)

                    test_loss += loss.item()

                    batch_time.update(time.time() - end)
                    losses.update(loss.item(), X.shape[0])
                    local_r2 = compute_r2(y_pred, y.to(device)).detach().cpu().numpy()
                    r2.update(local_r2, X.shape[0])

                    if iter % 1 == 0:
                        # print('Test Epoch', epoch, 'Batch', iter, 'Meta Test Loss', test_loss)
                        progress.display(iter + 1)
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


def get_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.exp_dir, 'model_best.pkl'))


def get_argparser():
    parser = argparse.ArgumentParser(description='Meta Learning')
    parser.add_argument('--data-dir', default='./data/block_sample_v3/', help='path to data')
    parser.add_argument('--pretrained-model', default='./checkpoints/best_model_synthetic_split_by_space_v2.pkl',
                        help='path to pretrained model')
    parser.add_argument('--exp-dir', default='./experiments/debug', help='path save experimental results')
    parser.add_argument('--num-workers', default=8, type=int, help='number of workers used in dataloader')
    parser.add_argument('--crop', default="corn", choices=["corn", "soybean"], help='crop category')
    parser.add_argument('--num-epochs', default=2, type=int, help='number of running epochs')
    parser.add_argument('--gpus', default=0, type=int, help='number of gpus')
    parser.add_argument('--seed', default=20, type=int, help='random seed number')
    parser.add_argument('--num-per-support', default=25, type=int, help='number of samples per support set')
    parser.add_argument('--num-per-query', default=75, type=int, help='number of samples per query set')
    parser.add_argument('--task-per-batch', default=16, type=int, help='number of tasks per batch')
    parser.add_argument('--num-tasks', default=0, type=int, help='number of tasks to set')
    parser.add_argument('--adapt-lr', default=0.001, type=float, help='adaptive learning rate')
    parser.add_argument('--meta-lr', default=0.001, type=float, help='meta learning rate')
    parser.add_argument('--adapt-steps', default=1, type=int, help='adaptive steps')
    parser.add_argument('--baseline-mode', default=3, type=int, choices=[0, 1, 2, 3],
                        help='baseline mode: '
                             '0-use train support as training set '
                             '1-use train support and train query as training set; '
                             '2-use train support, train query, test support as training set; '
                             '3-use train support, train query as training set, and finetune on test support')

    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    device = torch.device("cuda")
    main()
