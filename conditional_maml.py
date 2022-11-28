import shutil
import pandas as pd
import argparse
import time
from sklearn.cluster import KMeans
from utilities import AverageMeter, ProgressMeter, HelperFunctions, R2Loss
import numpy as np
import os
import torch
import json
import pickle
from models import CornYieldModel, LossFunctions
import learn2learn as l2l
from torch import nn, optim
from tensorboard_logger import configure, log_value
import warnings

warnings.filterwarnings("ignore")


def train(train_data, evaluate_fips, train_fips, epoch, iter_count=0, mode="init", exp_postfix="",
          best_train_r2=float("-inf")):
    lossfn = nn.MSELoss(reduction='mean')
    compute_r2 = R2Loss()
    helper = HelperFunctions()
    convert_index_corn = 0.429

    init_model_path = args.pretrained_model
    checkpoint = torch.load(init_model_path)
    model = CornYieldModel(num_features=19, hidden_size=64).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    maml = l2l.algorithms.MAML(model, lr=args.adapt_lr, first_order=False, allow_unused=True, allow_nograd=False)
    optimizer = optim.Adam(maml.parameters(), args.meta_lr)

    train_data_batch = get_batch_data(train_data, train_fips, args.task_per_batch)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    r2 = AverageMeter('R2', ':.4e')

    progress = ProgressMeter(len(train_data_batch), [batch_time, data_time, losses, r2],
                             prefix="Train [{}] Model Epoch: [{}] Cluster: [{}]".format(mode, epoch, iter_count))
    end = time.time()
    # train_task_loss = []

    inner_best_R2 = float('-inf')
    inner_best_model = None
    inner_best_train_task_r2 = []
    for inner_epoch in range(args.num_inner_epochs):
        print(f"Train inner-epoch [{inner_epoch}][{args.num_inner_epochs}]")
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
                progress.display(iter)

            r2.update(local_r2.item(), batch_tasks)
            batch_time.update(time.time() - end)
            losses.update(meta_train_loss.item(), batch_tasks)

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

    ckpt_name = 'model_epoch_' + str(epoch) + f'_cluster_{iter_count}.pkl'

    file_name = os.path.join(args.exp_dir, ckpt_name)
    save_checkpoint(args, epoch, mode, {
        'epoch': epoch,
        'state_dict': inner_best_model.state_dict(),
        'r2': inner_best_R2,
        'optimizer': optimizer.state_dict(),
    }, is_best, file_name)

    # return R2, train_task_loss
    return inner_best_R2, inner_best_train_task_r2


def test(test_data, test_fips, epoch, mode="init", cluster_dict=None, exp_postfix=""):
    lossfn = nn.MSELoss(reduction='mean')
    compute_r2 = R2Loss()
    helper = HelperFunctions()
    convert_index_corn = 0.429

    pred_arr = []
    gold_arr = []
    fips_id_arr = []
    year_id_arr = []
    model_dict = {}

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
                             prefix="Test Epoch: [{}]".format(epoch))

    end = time.time()
    for iter, batch in enumerate(test_data_batch):
        data_time.update(time.time() - end)
        meta_test_loss = 0.0

        batch_tasks = batch[0].shape[0]
        batch_pred = []
        batch_gold = []
        for i in range(batch_tasks):

            support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
                batch[0][i], batch[1][i], batch[2][i], batch[3][i], batch[4][i], batch[5][i], batch[6][i], \
                batch[7][i], batch[8][i], batch[9][i]

            iter_count = cluster_dict[int(support_fips[0])]
            best_ckpt_name = f"model_epoch_{epoch}_cluster_{iter_count}.pkl"
            best_maml = model_dict[best_ckpt_name]
            # print("Test with Model:", best_ckpt_name)
            best_learner = best_maml.clone()
            if len(gpu_ids) > 1:
                best_learner.module = torch.nn.DataParallel(best_learner.module, device_ids=gpu_ids)

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
            progress.display(iter)
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


def get_all_model_names(epoch):
    all_model_name_list = []
    iter_count = 0
    while iter_count < int(args.num_clusters):
        ckpt_name = f"model_epoch_{epoch}_cluster_{iter_count}.pkl"
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
                             prefix="Evaluate [{}] Model Epoch: [{}] Cluster: [{}]".format(mode, epoch, iter_count))

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
            progress.display(iter)
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
        df.to_csv(os.path.join(args.exp_dir, f"result_{mode}_epoch_{epoch}_cluster_{iter_count}_{exp_postfix}.csv"))
        helper.plot(pred_res, gold_res, args.exp_dir, f"plot_{mode}_epoch_{epoch}_cluster_{iter_count}_{exp_postfix}")

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


def get_cluster_model(data):
    input_list = []
    fips_list = []
    for iter, batch in enumerate(data.items()):
        batch = batch[1]
        support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
        batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9]
        input = torch.cat([support_x, query_x], dim=1).reshape(-1, support_x.shape[-2], support_x.shape[-1])
        fips = torch.cat([support_fips, query_fips], dim=1).reshape(-1, )
        input_list.append(input.numpy()[:1])
        fips_list.append(fips.numpy()[:1])

    input_array = np.array(input_list)
    input_array = input_array.reshape(input_array.shape[0], -1)
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(input_array)
    labels = kmeans.labels_

    dict = {int(fips_list[i]): int(labels[i]) for i in range(len(input_list))}
    with open(os.path.join(args.exp_dir, "train_clusters.json"), "w") as f:
        json.dump(dict, f)

    return kmeans, dict


def predict_cluster(kmeans, data):
    input_list = []
    fips_list = []
    for iter, batch in enumerate(data.items()):
        batch = batch[1]
        support_x, support_y, support_fips, support_dp, support_year, query_x, query_y, query_fips, query_dp, query_year = \
        batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9]
        input = torch.cat([support_x, query_x], dim=1).reshape(-1, support_x.shape[-2], support_x.shape[-1])
        fips = torch.cat([support_fips, query_fips], dim=1).reshape(-1, )
        input_list.append(input.numpy()[:1])
        fips_list.append(fips.numpy()[:1])

    input_array = np.array(input_list)
    input_array = input_array.reshape(input_array.shape[0], -1)
    labels = kmeans.predict(input_array)

    dict = {int(fips_list[i]): int(labels[i]) for i in range(len(input_list))}
    with open(os.path.join(args.exp_dir, "test_clusters.json"), "w") as f:
        json.dump(dict, f)

    return dict


def main():
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    with open(os.path.join(args.exp_dir, 'configs.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.test_only != 1:
        if os.path.exists(os.path.join(args.exp_dir, "train_clusters.json")):
            os.remove(os.path.join(args.exp_dir, "train_clusters.json"))

        if os.path.exists(os.path.join(args.exp_dir, "test_clusters.json")):
            os.remove(os.path.join(args.exp_dir, "test_clusters.json"))

    exp_postfix = f"support_size_{args.num_per_support}_query_size_{args.num_per_query}"
    configure(args.exp_dir)

    data = get_data(f"./data/fine_sample/{args.crop}_block_sample.pkl")
    train_data = data["train"]


    train_fips = list(train_data.keys())
    test_data = data["test"]
    test_fips = list(test_data.keys())

    kmeans, train_clusters = get_cluster_model(train_data)
    test_clusters = predict_cluster(kmeans, test_data)

    for epoch in range(args.num_epochs):
        if args.test_only == 1:
            phases = ["test"]
        else:
            phases = ["train", "test"]
        for phase in phases:
            if phase == "train":

                # train model for each cluster
                for cluster_id in range(args.num_clusters):
                    cluster_train_data = {}
                    cluster_train_fips = []
                    for this_fips in train_fips:
                        if train_clusters[this_fips] == cluster_id:
                            cluster_train_data[this_fips] = train_data[this_fips]
                            cluster_train_fips.append(this_fips)

                    _, train_task_loss = train(cluster_train_data, cluster_train_fips, cluster_train_fips, epoch,
                                               iter_count=cluster_id, mode="init", exp_postfix=exp_postfix)

            if phase == "test":
                test(test_data, test_fips, epoch, cluster_dict=test_clusters, mode="init", exp_postfix=exp_postfix)


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
    parser.add_argument('--data-path', default='./data/fine_sample/corn_block_sample.pkl', help='path to data')
    parser.add_argument('--pretrained-model', default='./checkpoints/best_model_synthetic_split_by_space_v2.pkl',
                        help='path to pretrained model')
    parser.add_argument('--exp-dir', default='./experiments/debug', help='path save experimental results')
    parser.add_argument('--num-workers', default=8, type=int, help='number of workers used in dataloader')
    parser.add_argument('--crop', default="corn", choices=["corn", "soybean"], help='crop category')
    parser.add_argument('--num-epochs', default=20, type=int, help='number of running epochs')
    parser.add_argument('--num-inner-epochs', default=1, type=int, help='number of inner running epochs')
    parser.add_argument('--num-clusters', default=2, type=int, help='number of clusters')
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
