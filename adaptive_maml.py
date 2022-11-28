import argparse
import shutil
import time
import warnings
import learn2learn as l2l
import pandas as pd
from tensorboard_logger import configure, log_value
from torch import nn, optim
from models import CornYieldModel
import numpy as np
import os
import torch
import json
import pickle
from utilities import AverageMeter, ProgressMeter, R2Loss, HelperFunctions

warnings.filterwarnings("ignore")


def train(train_data, evaluate_fips, train_fips, epoch, iter_count=0, mode="init", exp_postfix="",
          best_train_r2=float("-inf")):
    lossfn = nn.MSELoss(reduction='mean')
    compute_r2 = R2Loss()
    helper = HelperFunctions()
    convert_index_corn = 0.429

    if mode == "init":
        if epoch == 0:
            init_model_path = args.pretrained_model
            checkpoint = torch.load(init_model_path)
            model = CornYieldModel(num_features=19, hidden_size=64).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            ckpt_name = 'model_epoch_' + str(epoch - 1) + f'_hard_iter_{args.max_iter_num - 1}.pkl'
            init_model_path = os.path.join(args.exp_dir, ckpt_name)
            checkpoint = torch.load(init_model_path)
            model = CornYieldModel(num_features=19, hidden_size=64).to(device)
            model.load_state_dict(checkpoint['state_dict'])
    else:
        if iter_count == 0:
            ckpt_name = 'model_epoch_' + str(epoch) + '_init.pkl'
            init_model_path = os.path.join(args.exp_dir, ckpt_name)
            checkpoint = torch.load(init_model_path)
            model = CornYieldModel(num_features=19, hidden_size=64).to(device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            ckpt_name = 'model_epoch_' + str(epoch) + f'_hard_iter_{iter_count - 1}.pkl'
            init_model_path = os.path.join(args.exp_dir, ckpt_name)
            checkpoint = torch.load(init_model_path)
            model = CornYieldModel(num_features=19, hidden_size=64).to(device)
            model.load_state_dict(checkpoint['state_dict'])

    maml = l2l.algorithms.MAML(model, lr=args.adapt_lr, first_order=False, allow_unused=True, allow_nograd=False)
    optimizer = optim.Adam(maml.parameters(), args.meta_lr)

    train_data_batch = get_batch_data(train_data, train_fips, args.task_per_batch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    r2 = AverageMeter('R2', ':.4e')

    if mode == "init":
        progress = ProgressMeter(len(train_data_batch), [batch_time, data_time, losses, r2],
                                 prefix="Train [{}] Model Epoch: [{}] Batch".format(mode, epoch))
    else:
        progress = ProgressMeter(len(train_data_batch), [batch_time, data_time, losses, r2],
                                 prefix="Train [{}] Model Epoch: [{}] Level: [{}] Batch:".format(mode, epoch,
                                                                                                 iter_count))
    end = time.time()
    inner_best_R2 = float('-inf')
    inner_best_model = None
    inner_best_train_task_r2 = []
    for inner_epoch in range(args.num_inner_epochs):
        print(f"Train inner-epoch [{inner_epoch + 1}] out of [{args.num_inner_epochs}]")
        pred_arr = []
        gold_arr = []
        fips_id_arr = []
        year_id_arr = []

        train_task_r2 = []
        for iter, batch in enumerate(train_data_batch):
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
                # train_task_loss.append(query_loss.detach().cpu().numpy())
                query_r2 = compute_r2(query_preds, query_y.to(device)).detach().cpu().numpy()
                train_task_r2.append(query_r2.item())

                pred_arr.append(query_preds)
                gold_arr.append(query_y)
                fips_id_arr.append(query_fips)
                year_id_arr.append(query_year)

                batch_pred.append(query_preds)
                batch_gold.append(query_y)

                optimizer.zero_grad()
                query_loss.backward()
                optimizer.step()

            batch_pred = torch.cat(batch_pred)
            batch_gold = torch.cat(batch_gold)

            local_r2 = compute_r2(batch_pred, batch_gold.to(device)).detach().cpu().numpy()
            meta_train_loss = meta_train_loss / batch_tasks

            if iter % 1 == 0:
                progress.display(iter + 1)

            if inner_epoch == args.num_inner_epochs - 1:
                # record the last inner epoch
                r2.update(local_r2.item(), batch_tasks)
                batch_time.update(time.time() - end)
                losses.update(meta_train_loss.item(), batch_tasks)

        if mode == "init":
            # group predicted results
            # init model use the whole data so just compute R2
            pred_arr = torch.cat(pred_arr).cpu().squeeze()
            gold_arr = torch.cat(gold_arr).cpu().squeeze()
            fips_id_arr = torch.cat(fips_id_arr).cpu().squeeze()
            year_id_arr = torch.cat(year_id_arr).cpu().squeeze()

            pred_res = helper.Z_norm_reverse(pred_arr, helper.scalar[0]) * convert_index_corn
            gold_res = helper.Z_norm_reverse(gold_arr, helper.scalar[0]) * convert_index_corn

            df = pd.DataFrame(
                {"fips_id": fips_id_arr.tolist(), "year": year_id_arr.tolist(), "pred": pred_res.tolist(),
                 "gold": gold_res.tolist()})
            df = df.groupby(['fips_id', 'year'], as_index=False).mean()
            df = df.sort_values(["fips_id", "year"], ascending=(True, True))
            pred_res, gold_res = np.array(df["pred"]), np.array(df["gold"])
            pred_res = torch.from_numpy(pred_res).to(device)
            gold_res = torch.from_numpy(gold_res).to(device)
            R2 = compute_r2(pred_res, gold_res).detach().cpu().numpy()
            print()  # add empty printing for beautifying outputs
        else:
            R2, df, pred_res, gold_res = evaluate(train_data, evaluate_fips, maml, mode, epoch, iter_count,
                                                  exp_postfix=exp_postfix, use_iter=True)

        ## inner loop best R2
        if R2.item() > inner_best_R2:
            inner_best_R2 = R2.item()
            inner_best_model = model
            inner_best_train_task_r2 = train_task_r2

    # save model
    if inner_best_R2 > best_train_r2:
        is_best = True
    else:
        is_best = False
    if mode == "init":
        ckpt_name = 'model_epoch_' + str(epoch) + '_init.pkl'
        is_best = False  # do not save init best model
    elif mode == "easy":
        ckpt_name = 'model_epoch_' + str(epoch) + f'_easy_iter_{iter_count}.pkl'
    else:
        ckpt_name = 'model_epoch_' + str(epoch) + f'_hard_iter_{iter_count}.pkl'

    file_name = os.path.join(args.exp_dir, ckpt_name)
    save_checkpoint(args, epoch, mode, {
        'epoch': epoch,
        'state_dict': inner_best_model.state_dict(),
        'r2': inner_best_R2,
        'optimizer': optimizer.state_dict(),
    }, is_best, file_name)

    # return R2, train_task_loss
    return inner_best_R2, inner_best_train_task_r2


def adaptive_tree_test(test_data, test_fips, epoch, mode="init", exp_postfix=""):
    lossfn = nn.MSELoss(reduction='mean')
    compute_r2 = R2Loss()
    helper = HelperFunctions()
    convert_index_corn = 0.429

    with open(os.path.join(args.exp_dir, "split_threshold.txt"), "r") as f:
        lines = f.readlines()
    split_dict = {}
    for line in lines:
        epoch = int(line.strip().split(" ")[0])
        iter_count = int(line.strip().split(" ")[1])
        split_threshold = float(line.strip().split(" ")[2])

        if epoch not in split_dict:
            split_dict[epoch] = [[iter_count, split_threshold]]
        else:
            split_dict[epoch].append([iter_count, split_threshold])

    pred_arr = []
    gold_arr = []
    fips_id_arr = []
    year_id_arr = []
    model_dict = {}

    if mode == "easy":
        init_ckpt_name = 'model_epoch_' + str(epoch) + f'_easy_best.pkl'
        init_model_path = os.path.join(args.exp_dir, init_ckpt_name)
    elif mode == "hard":
        init_ckpt_name = 'model_epoch_' + str(epoch) + f'_hard_best.pkl'
        init_model_path = os.path.join(args.exp_dir, init_ckpt_name)
    else:
        init_ckpt_name = 'model_epoch_' + str(epoch) + f'_init.pkl'
        init_model_path = os.path.join(args.exp_dir, init_ckpt_name)

    checkpoint = torch.load(init_model_path)
    model = CornYieldModel(num_features=19, hidden_size=64).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    maml = l2l.algorithms.MAML(model, lr=args.adapt_lr, first_order=False, allow_unused=True, allow_nograd=False)
    model_dict[init_ckpt_name] = maml

    ckpt_name_list = get_all_model_names(epoch)
    for ckpt_name in ckpt_name_list:
        checkpoint = torch.load(os.path.join(args.exp_dir, ckpt_name))
        model = CornYieldModel(num_features=19, hidden_size=64).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        maml = l2l.algorithms.MAML(model, lr=args.adapt_lr, first_order=False, allow_unused=True, allow_nograd=False)
        model_dict[ckpt_name] = maml

    test_data_batch = get_batch_data(test_data, test_fips, args.task_per_batch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    r2 = AverageMeter('R2', ':.4e')
    progress = ProgressMeter(len(test_data_batch), [batch_time, data_time, losses, r2],
                             prefix="Test Epoch: [{}] Batch".format(epoch))

    end = time.time()
    for iter, batch in enumerate(test_data_batch):
        data_time.update(time.time() - end)
        meta_test_loss = 0.0

        batch_tasks = batch[0].shape[0]
        batch_pred = []
        batch_gold = []
        for i in range(batch_tasks):
            # select best model based on loss for this data by retrieving the tree paths
            maml = model_dict[init_ckpt_name]
            learner = maml.clone()
            if len(gpu_ids) > 1:
                learner.module = torch.nn.DataParallel(learner.module, device_ids=gpu_ids)

            support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i], batch[5][i], batch[6][i], \
                batch[7][i], batch[8][i], batch[9][i]

            support_r2 = 0.5
            for _ in range(args.adapt_steps + args.adapt_steps_extra):
                support_preds = learner(support_x.to(device)).squeeze(2)
                support_loss = lossfn(support_preds, support_y.to(device))
                learner.adapt(support_loss, allow_unused=True, allow_nograd=True)
                support_r2 = compute_r2(support_preds, support_y.to(device)).item()

            this_iter_count = 0
            best_ckpt_name = init_ckpt_name
            while this_iter_count < int(args.max_iter_num):
                best_ckpt_name = get_checkpoint_name(epoch, split_dict, support_r2, this_iter_count=this_iter_count)
                if "easy" in best_ckpt_name: break

                maml = model_dict[best_ckpt_name]
                learner = maml.clone()
                if len(gpu_ids) > 1:
                    learner.module = torch.nn.DataParallel(learner.module, device_ids=gpu_ids)

                support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                    batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i], batch[5][i], batch[6][i], \
                    batch[7][i], batch[8][i], batch[9][i]

                support_r2 = 0
                for _ in range(args.adapt_steps + args.adapt_steps_extra):
                    support_preds = learner(support_x.to(device)).squeeze(2)
                    support_loss = lossfn(support_preds, support_y.to(device))
                    learner.adapt(support_loss, allow_unused=True, allow_nograd=True)
                    support_r2 = compute_r2(support_preds, support_y.to(device)).item()

                this_iter_count += 1

            best_maml = model_dict[best_ckpt_name]
            # print("Test with Model:", best_ckpt_name)
            best_learner = best_maml.clone()
            if len(gpu_ids) > 1:
                best_learner.module = torch.nn.DataParallel(best_learner.module, device_ids=gpu_ids)

            support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i], batch[5][i], batch[6][i], \
                batch[7][i], batch[8][i], batch[9][i]

            for _ in range(args.adapt_steps + args.adapt_steps_extra):
                support_preds = best_learner(support_x.to(device)).squeeze(2)
                support_loss = lossfn(support_preds, support_y.to(device))
                best_learner.adapt(support_loss, allow_unused=True, allow_nograd=True)

            with torch.no_grad():
                query_preds = best_learner(query_x.to(device)).squeeze(2)

            query_loss = lossfn(query_preds, query_y.to(device))
            meta_test_loss += query_loss

            pred_arr.append(query_preds)
            gold_arr.append(query_y)
            fips_id_arr.append(query_fips)
            year_id_arr.append(query_year)

            batch_pred.append(query_preds)
            batch_gold.append(query_y)

        meta_test_loss = meta_test_loss / batch_tasks

        batch_pred = torch.cat(batch_pred)
        batch_gold = torch.cat(batch_gold)
        local_r2 = compute_r2(batch_pred, batch_gold.to(device)).detach().cpu().numpy()
        r2.update(local_r2, batch_tasks)
        batch_time.update(time.time() - end)
        losses.update(meta_test_loss.item(), batch_tasks)

        if iter % 1 == 0:
            progress.display(iter + 1)
            step = iter + len(test_data_batch) * epoch
            log_value('test/epoch', epoch, step)
            log_value('test/loss', progress.meters[2].avg, step)

    pred_arr = torch.cat(pred_arr).cpu().squeeze()
    gold_arr = torch.cat(gold_arr).cpu().squeeze()
    fips_id_arr = torch.cat(fips_id_arr).cpu().squeeze()
    year_id_arr = torch.cat(year_id_arr).cpu().squeeze()

    pred_res = helper.Z_norm_reverse(pred_arr, helper.scalar[0]) * convert_index_corn
    gold_res = helper.Z_norm_reverse(gold_arr, helper.scalar[0]) * convert_index_corn

    df = pd.DataFrame(
        {"fips_id": fips_id_arr.tolist(), "year": year_id_arr.tolist(), "pred": pred_res.tolist(),
         "gold": gold_res.tolist()})
    df = df.groupby(['fips_id', 'year'], as_index=False).mean()
    df = df.sort_values(["fips_id", "year"], ascending=(True, True))
    pred_res, gold_res = np.array(df["pred"]), np.array(df["gold"])
    pred_res = torch.from_numpy(pred_res).to(device)
    gold_res = torch.from_numpy(gold_res).to(device)
    R2 = compute_r2(pred_res, gold_res).detach().cpu().numpy()

    df.to_csv(os.path.join(args.exp_dir, f"result_test_epoch_{epoch}_{exp_postfix}.csv"))
    helper.plot(pred_res, gold_res, args.exp_dir, f"plot_test_epoch_{epoch}_{exp_postfix}")

    if epoch % 1 == 0:
        log_value('test/r2', R2, epoch)
    print(f"Test Best Model Epoch [{epoch}]: R2 {R2.item()}")
    print("----------------------------------------------\n")


def get_best_model(epoch, split_dict, value):
    # this function is used to return the best model path given a loss value
    split_data = split_dict[epoch]
    this_iter_count = 0
    this_split_threshold = 0.5
    flag = float('inf')
    for split_list in split_data:
        iter_count, split_threshold = split_list
        if abs(split_threshold - value) < flag:
            flag = abs(split_threshold - value)
            this_iter_count = iter_count
            this_split_threshold = split_threshold
    if value > this_split_threshold:
        ckpt_name = f"model_epoch_{epoch}_easy_iter_{this_iter_count}.pkl"
    else:
        ckpt_name = f"model_epoch_{epoch}_hard_iter_{this_iter_count}.pkl"
    print(f"Load Best Model [{ckpt_name}] Epoch [{epoch}]")
    return ckpt_name


def get_checkpoint_name(epoch, split_dict, value, this_iter_count=0):
    this_split_threshold = 0.5
    for iter_count, threshold in split_dict[epoch]:
        this_split_threshold = threshold
        if iter_count == this_iter_count:
            break
    if value > this_split_threshold:
        ckpt_name = f"model_epoch_{epoch}_easy_iter_{this_iter_count}.pkl"
    else:
        ckpt_name = f"model_epoch_{epoch}_hard_iter_{this_iter_count}.pkl"
    return ckpt_name


def get_all_model_names(epoch):
    all_model_name_list = []
    iter_count = 0
    while iter_count < int(args.max_iter_num):
        for mode in ["easy", "hard"]:
            ckpt_name = f"model_epoch_{epoch}_{mode}_iter_{iter_count}.pkl"
            all_model_name_list.append(ckpt_name)
        iter_count += 1

    return all_model_name_list


def evaluate(data, fips, maml, mode, epoch, iter_count, exp_postfix="", use_iter=False):
    lossfn = nn.MSELoss(reduction='mean')
    compute_r2 = R2Loss()
    helper = HelperFunctions()
    convert_index_corn = 0.429

    pred_arr = []
    gold_arr = []
    fips_id_arr = []
    year_id_arr = []

    test_data_batch = get_batch_data(data, fips, args.task_per_batch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    r2 = AverageMeter('R2', ':.4e')
    progress = ProgressMeter(len(test_data_batch), [batch_time, data_time, losses, r2],
                             prefix="Evaluate [{}] Model Epoch: [{}] Iter: [{}] Batch:".format(mode, epoch, iter_count))

    end = time.time()
    for iter, batch in enumerate(test_data_batch):
        data_time.update(time.time() - end)
        meta_test_loss = 0.0

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

            with torch.no_grad():
                query_preds = learner(query_x.to(device)).squeeze(2)

            query_loss = lossfn(query_preds, query_y.to(device))
            meta_test_loss += query_loss

            pred_arr.append(query_preds)
            gold_arr.append(query_y)
            fips_id_arr.append(query_fips)
            year_id_arr.append(query_year)

            batch_pred.append(query_preds)
            batch_gold.append(query_y)

        meta_test_loss = meta_test_loss / batch_tasks

        batch_pred = torch.cat(batch_pred)
        batch_gold = torch.cat(batch_gold)
        local_r2 = compute_r2(batch_pred, batch_gold.to(device)).detach().cpu().numpy()
        r2.update(local_r2, batch_tasks)
        batch_time.update(time.time() - end)
        losses.update(meta_test_loss.item(), batch_tasks)

        if iter % 1 == 0:
            progress.display(iter + 1)
            step = iter + len(test_data_batch) * epoch
            log_value(f'{mode}_{iter_count}/epoch', epoch, step)
            log_value(f'{mode}_{iter_count}/loss', progress.meters[2].avg, step)

    pred_arr = torch.cat(pred_arr).cpu().squeeze()
    gold_arr = torch.cat(gold_arr).cpu().squeeze()
    fips_id_arr = torch.cat(fips_id_arr).cpu().squeeze()
    year_id_arr = torch.cat(year_id_arr).cpu().squeeze()

    pred_res = helper.Z_norm_reverse(pred_arr, helper.scalar[0]) * convert_index_corn
    gold_res = helper.Z_norm_reverse(gold_arr, helper.scalar[0]) * convert_index_corn

    df = pd.DataFrame(
        {"fips_id": fips_id_arr.tolist(), "year": year_id_arr.tolist(), "pred": pred_res.tolist(),
         "gold": gold_res.tolist()})
    df = df.groupby(['fips_id', 'year'], as_index=False).mean()
    df = df.sort_values(["fips_id", "year"], ascending=(True, True))
    pred_res, gold_res = np.array(df["pred"]), np.array(df["gold"])
    pred_res = torch.from_numpy(pred_res).to(device)
    gold_res = torch.from_numpy(gold_res).to(device)
    R2 = compute_r2(pred_res, gold_res).detach().cpu().numpy()

    if use_iter:
        df.to_csv(os.path.join(args.exp_dir, f"result_{mode}_epoch_{epoch}_iter_{iter_count}_{exp_postfix}.csv"))
        helper.plot(pred_res, gold_res, args.exp_dir, f"plot_{mode}_epoch_{epoch}_iter_{iter_count}_{exp_postfix}")
    else:
        df.to_csv(os.path.join(args.exp_dir, f"result_{mode}_epoch_{epoch}_{exp_postfix}.csv"))
        helper.plot(pred_res, gold_res, args.exp_dir, f"plot_{mode}_epoch_{epoch}_{exp_postfix}")

    if epoch % 1 == 0:
        log_value(f'{mode}_{iter_count}/r2', R2, epoch)

    return R2, df, pred_res, gold_res


def get_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


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
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    with open(os.path.join(args.exp_dir, 'configs.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.test_only != 1:
        if os.path.exists(os.path.join(args.exp_dir, "tasks_easy.txt")):
            os.remove(os.path.join(args.exp_dir, "tasks_easy.txt"))
        if os.path.exists(os.path.join(args.exp_dir, "tasks_hard.txt")):
            os.remove(os.path.join(args.exp_dir, "tasks_hard.txt"))
        if os.path.exists(os.path.join(args.exp_dir, "split_threshold.txt")):
            os.remove(os.path.join(args.exp_dir, "split_threshold.txt"))

    exp_postfix = f"support_size_{args.num_per_support}_query_size_{args.num_per_query}"
    configure(args.exp_dir)

    data = get_data(f"./data/fine_sample/{args.crop}_block_sample.pkl")
    train_data = data["train"]

    train_fips = list(train_data.keys())
    test_data = data["test"]
    test_fips = list(test_data.keys())

    for epoch in range(args.num_epochs):
        if args.test_only == 1:
            phases = ["test"]
        else:
            phases = ["train", "test"]
        for phase in phases:
            if phase == "train":
                _, train_task_loss = train(train_data, train_fips, train_fips, epoch, mode="init",
                                           exp_postfix=exp_postfix)
                train_fips_easy, train_fips_hard = get_split_threshold(train_fips, train_task_loss, epoch, iter_count=0)

                iter_count = 0
                best_R2_easy = float('-inf')
                best_R2_hard = float('-inf')
                while len(train_fips_hard) > 0 and iter_count < args.max_iter_num:
                    # R2_easy, _ = train(train_data, train_fips, train_fips_easy, epoch, iter_count=iter_count, mode="easy", exp_postfix=exp_postfix, best_train_r2=best_R2_easy)
                    R2_easy, _ = train(train_data, train_fips_easy, train_fips_easy, epoch, iter_count=iter_count,
                                       mode="easy", exp_postfix=exp_postfix, best_train_r2=best_R2_easy)
                    # R2_hard, train_task_loss = train(train_data, train_fips, train_fips_hard, epoch, iter_count=iter_count, mode="hard", exp_postfix=exp_postfix, best_train_r2=best_R2_hard)
                    R2_hard, train_task_loss = train(train_data, train_fips_hard, train_fips_hard, epoch,
                                                     iter_count=iter_count, mode="hard", exp_postfix=exp_postfix,
                                                     best_train_r2=best_R2_hard)
                    if iter_count < args.max_iter_num - 1:
                        train_fips_easy, train_fips_hard = get_split_threshold(train_fips_hard, train_task_loss, epoch,
                                                                               iter_count=iter_count + 1)
                    iter_count += 1

                    if R2_easy > best_R2_easy:
                        best_R2_easy = R2_easy
                    if R2_hard > best_R2_hard:
                        best_R2_hard = R2_hard
            if phase == "test":
                adaptive_tree_test(test_data, test_fips, epoch, mode="init", exp_postfix=exp_postfix)


def get_split_threshold(train_fips, train_loss, epoch, iter_count=0, version=1):
    ranked_loss = [x for x, y in sorted(zip(train_loss, train_fips))]
    ranked_fips = [y for x, y in sorted(zip(train_loss, train_fips))]

    if version == 0:
        split = 0
        threshold = 1 - 0.5 ** (iter_count + 1)
        for k in range(1, len(ranked_loss) - 1):
            if ranked_loss[k] > threshold:
                split = k
                break
    else:
        var_list = []
        quantile1 = int(len(ranked_loss) * 0.35)
        quantile2 = int(len(ranked_loss) * 0.65)
        for k in range(len(ranked_loss)):
            hard_loss = np.array(ranked_loss[:k])
            easy_loss = np.array(ranked_loss[k:])
            total_var = np.var(hard_loss) + np.var(easy_loss)
            var_list.append(total_var)
        if len(var_list) == 0:
            return 0
        threshold = min(var_list[quantile1:quantile2 + 1])
        split = var_list.index(threshold)

    easy_fips = ranked_fips[split:]
    hard_fips = ranked_fips[:split]
    split_loss = ranked_loss[split]

    with open(os.path.join(args.exp_dir, "tasks_easy.txt"), "a") as f:
        text_str = f"epoch {epoch} iter {iter_count} threshold {split_loss}:" + " ".join(str(i) for i in easy_fips)
        f.write(f"{text_str}\n")
    with open(os.path.join(args.exp_dir, "tasks_hard.txt"), "a") as f:
        text_str = f"epoch {epoch} iter {iter_count} threshold {split_loss}:" + " ".join(str(i) for i in hard_fips)
        f.write(f"{text_str}\n")
    with open(os.path.join(args.exp_dir, "split_threshold.txt"), "a") as f:
        text_str = f"{epoch} {iter_count} {split_loss}"
        f.write(f"{text_str}\n")
    print(f"Easy/Hard Splitting (R2) Threshold Epoch [{epoch}] Iter [{iter_count}]:", split_loss)
    print("Easy Tasks number:", len(easy_fips))
    print("Hard Tasks number:", len(hard_fips))
    print()

    return easy_fips, hard_fips


def save_checkpoint(args, epoch, mode, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        ckpt_name = 'model_epoch_' + str(epoch) + f'_{mode}_best.pkl'
        shutil.copyfile(filename, os.path.join(args.exp_dir, ckpt_name))


def get_device():
    gpu_ids = [int(i) for i in args.gpus.split(",")]
    if "debug" in args.exp_dir:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids[0]}"
    device = torch.device("cuda")
    print(f"use device {gpu_ids} gpus")
    return device, gpu_ids


def get_argparser():
    parser = argparse.ArgumentParser(description='Meta Learning')
    parser.add_argument('--data-path', default='./data/fine_sample/soybean_block_sample.pkl', help='path to data')
    parser.add_argument('--pretrained-model', default='./checkpoints/best_model_synthetic_split_by_space_v2.pkl',
                        help='path to pretrained model')
    parser.add_argument('--exp-dir', default='./experiments/debug', help='path save experimental results')
    parser.add_argument('--num-workers', default=8, type=int, help='number of workers used in dataloader')
    parser.add_argument('--crop', default="corn", choices=["corn", "soybean"], help='crop category')
    parser.add_argument('--num-epochs', default=20, type=int, help='number of running epochs')
    parser.add_argument('--num-inner-epochs', default=1, type=int, help='number of inner running epochs')
    parser.add_argument('--gpus', default='0', type=str, help='specified gpus')
    parser.add_argument('--seed', default=20, type=int, help="random seed number")
    parser.add_argument('--num-per-support', default=25, type=int, help='number of samples per support set')
    parser.add_argument('--num-per-query', default=75, type=int, help='number of samples per query set')
    parser.add_argument('--task-per-batch', default=32, type=int, help='number of tasks per batch')
    parser.add_argument('--num-tasks', default=64, type=int, help='number of tasks to set')
    parser.add_argument('--adapt-lr', default=0.001, type=float, help='adaptive learning rate')
    parser.add_argument('--meta-lr', default=0.001, type=float, help='meta learning rate')
    parser.add_argument('--adapt-steps', default=1, type=int, help='adaptive steps')
    parser.add_argument('--adapt-steps-extra', default=1, type=int, help='addictive adaptive steps in test finetune')
    parser.add_argument('--max-iter-num', default=3, type=int, help='the maximum iteration in each epoch')
    parser.add_argument('--test-only', default=0, type=int, help='only run test if set 1')

    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    device, gpu_ids = get_device()
    main()
