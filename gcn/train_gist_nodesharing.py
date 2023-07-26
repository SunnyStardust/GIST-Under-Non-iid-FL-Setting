import os

import dgl
from matplotlib import pyplot as plt

os.environ["DGL_REPO"] = "http://data.dgl.ai/"
import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data, LegacyTUDataset
from gcn import GCN, GraphGCN
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import train_test_split
import random
from random import choices
import json

def _randChunk(graphs, num_client, overlap, seed=None):
    # random.seed(seed)
    # np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum/num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i*minSize:(i+1)*minSize])
        for g in graphs[num_client*minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks

def split_data(graphs, train=None, test=None, shuffle=True, seed=None):
    y = torch.cat([torch.tensor([graph[1]]) for graph in graphs])
    graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, stratify=y, shuffle=shuffle, random_state=seed)
    return graphs_tv, graphs_test

def one_hot(graphs, max_num, device):
    for entry in graphs:
        g = entry[0]
        degs = []
        res = []
        for i in range(g.num_nodes()):
            degs.append(((g.all_edges()[0]==i).sum(dim=0)).item())
        for i in degs:
            tmp = torch.zeros(max_num)
            tmp[i] = 1
            res.append(tmp)
        g.ndata['feat'] = torch.stack(res).to(device)

def draw_acc_curve(record, save_root, save_filename):
    acc_train = np.array(record)[:, 0]
    acc_test = np.array(record)[:, 1]
    plt.title("Validation Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(len(acc_test)), acc_train, 'red', label="train")
    plt.plot(range(len(acc_test)), acc_test, 'orange', label="test")
    # plt.ylim([0.5, 1.0])
    plt.legend()
    np.save(os.path.join(save_root, 'record_' + save_filename), record)
    plt.savefig(os.path.join(save_root, save_filename + '.jpg'))
    plt.close()


def evaluate(model, loaders):
    model.eval()
    a = 0
    b = 0
    with torch.no_grad():
        for loader in loaders:
            for batched_graph, labels in loader:
                bnn = batched_graph.batch_num_nodes()
                bne = batched_graph.batch_num_edges()
                batched_graph = dgl.remove_self_loop(batched_graph)
                batched_graph = dgl.add_self_loop(batched_graph)
                batched_graph.set_batch_num_edges(bne)
                batched_graph.set_batch_num_nodes(bnn)
                pred = model(batched_graph, batched_graph.ndata['feat'].float())
                labels = labels.long()
                a += (pred.argmax(1) == labels).sum().item()
                b += len(labels)
        print(a/b)
        return a / b


def main(args):
    # convert boolean type for args
    assert args.use_ist in ['True', 'False'], ["Only True or False for use_ist, get ",
                                               args.use_ist]
    assert args.split_input in ['True', 'False'], ["Only True or False for split_input, get ",
                                                   args.split_input]
    assert args.split_output in ['True', 'False'], ["Only True or False for split_output, get ",
                                                   args.split_output]
    assert args.self_loop in ['True', 'False'], ["Only True or False for self_loop, get ",
                                                 args.self_loop]
    assert args.use_layernorm in ['True', 'False'], ["Only True or False for use_layernorm, get ",
                                                     args.use_layernorm]
    assert args.non_iid in ['True', 'False'], ["Only True or False for non_iid, get ",
                                               args.non_iid]
    use_ist = (args.use_ist == 'True')
    split_input = (args.split_input == 'True')
    split_output = (args.split_output == 'True')
    self_loop = (args.self_loop == 'True')
    use_layernorm = (args.use_layernorm == 'True')
    non_iid = (args.non_iid == 'True')

    # make sure hidden layer is the correct shape
    assert (args.n_hidden % args.num_subnet) == 0
    global t0
    data = LegacyTUDataset(args.dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs = [(dgl.to_simple(x[0]).to(device), x[1].to(device)) for x in data]

    # Handle IMDB-BINARY, where no features are given
    if args.dataset == "IMDB-BINARY":
        one_hot(graphs, 136, device)
    
    # if not non_iid:
    #     random.shuffle(graphs)
    in_feats = len(graphs[0][0].ndata['feat'][0])

    avg_best_train = 0
    avg_best_test = 0
    training_spec = {"runs": []}
    for run in range(1, 4):
        print(run, "-------------------------")
        chunks = _randChunk(graphs, args.num_subnet, False)
        loaders = []
        for chunk in chunks:
            ds_train, ds_test = split_data(chunk, train=0.8, test=0.2, shuffle=True)
            train_loader = GraphDataLoader(ds_train, batch_size=10, drop_last=False)
            test_loader = GraphDataLoader(ds_test, batch_size=10, drop_last=False)
            loaders.append((train_loader, test_loader))

        # create GCN model
        model = GraphGCN(
            in_feats, args.n_hidden, data.num_labels, args.n_layers, F.relu,
            args.dropout, use_layernorm)
        model = model.to(device)
        loss_fcn = torch.nn.CrossEntropyLoss()

        # initialize graph
        dur = []
        record = []
        sub_models = []
        opt_list = []
        sub_dict_list = []
        main_dict = None
        for epoch in range(args.n_epochs):
            print(epoch)
            if epoch >= 3:
                t0 = time.time()
            if use_ist:
                model.eval()
                # IST training:
                # Distribute parameter to sub networks
                num_subnet = args.num_subnet
                if (epoch % args.iter_per_site) == 0.:
                    main_dict = model.state_dict()
                    feats_idx = []  # store all layer indices within a single list

                    # create input partition
                    if split_input:
                        feats_idx.append(torch.chunk(torch.concat([torch.randperm(in_feats), torch.randperm(in_feats)]), num_subnet))
                    else:
                        feats_idx.append(None)

                    # create hidden layer partitions
                    for i in range(1, args.n_layers):
                        feats_idx.append(torch.chunk(torch.concat([torch.randperm(args.n_hidden), torch.randperm(args.n_hidden)]), num_subnet))

                    # create output layer partitions
                    if split_output:
                        feats_idx.append(torch.chunk(torch.concat([torch.randperm(args.n_hidden), torch.randperm(args.n_hidden)]), num_subnet))
                    else:
                        feats_idx.append(None)

                for subnet_id in range(args.num_subnet):
                    if (epoch % args.iter_per_site) == 0.:
                        # create the sub model to train
                        sub_model = GraphGCN(
                            in_feats, args.n_hidden, data.num_labels,
                            args.n_layers, F.relu, args.dropout, use_layernorm,
                            split_input, split_output, args.num_subnet / 2)
                        sub_model = sub_model.to(device)
                        sub_dict = main_dict.copy()

                        # split input params
                        if split_input:
                            idx = feats_idx[0][subnet_id]
                            sub_dict['layers.0.weight'] = main_dict['layers.0.weight'][idx, :]

                        # split hidden params (and output params)
                        for i in range(1, args.n_layers + 1):
                            if i == args.n_layers and not split_output:
                                pass  # params stay the same
                            else:
                                idx = feats_idx[i][subnet_id]
                                sub_dict[f'layers.{i - 1}.weight'] = sub_dict[f'layers.{i - 1}.weight'][:, idx]
                                sub_dict[f'layers.{i - 1}.bias'] = main_dict[f'layers.{i - 1}.bias'][idx]
                                sub_dict[f'layers.{i}.weight'] = main_dict[f'layers.{i}.weight'][idx, :]
                        # for k in sub_dict.keys():
                        #     print(k, len(sub_dict[k]))

                        # use a lr scheduler
                        curr_lr = args.lr
                        if epoch >= int(args.n_epochs * 0.5):
                            curr_lr /= 10
                        if epoch >= int(args.n_epochs * 0.75):
                            curr_lr /= 10

                        # import params into subnet for training
                        sub_model.load_state_dict(sub_dict)
                        sub_models.append(sub_model)
                        sub_models = sub_models[-num_subnet:]
                        optimizer = torch.optim.Adam(
                            sub_model.parameters(), lr=curr_lr,
                            weight_decay=args.weight_decay)
                        opt_list.append(optimizer)
                        opt_list = opt_list[-num_subnet:]
                    else:
                        sub_model = sub_models[subnet_id]
                        optimizer = opt_list[subnet_id]

                    # train a sub network
                    sub_model.train()
                    epoch_loss = 0
                    for batched_graph, labels in loaders[subnet_id][0]:
                        optimizer.zero_grad()
                        bnn = batched_graph.batch_num_nodes()
                        bne = batched_graph.batch_num_edges()
                        batched_graph = dgl.remove_self_loop(batched_graph)
                        batched_graph = dgl.add_self_loop(batched_graph)
                        batched_graph.set_batch_num_edges(bne)
                        batched_graph.set_batch_num_nodes(bnn)
                        labels = labels.long()
                        features = batched_graph.ndata['feat'].float()
                        if split_input:
                            model_input = features[:, feats_idx[0][subnet_id]]
                        else:
                            model_input = features
                        pred = sub_model(batched_graph, model_input)
                        loss = loss_fcn(pred, labels)
                        epoch_loss += loss.detach().item()
                        loss.backward()
                        optimizer.step()

                    # save sub model parameter
                    if (
                            ((epoch + 1) % args.iter_per_site == 0.)
                            or (epoch == args.n_epochs - 1)):
                        sub_dict = sub_model.state_dict()
                        sub_dict_list.append(sub_dict)
                        sub_dict_list = sub_dict_list[-num_subnet:]

                # Merge parameter to main network:
                # force aggregation if training about to end
                if (
                        ((epoch + 1) % args.iter_per_site == 0.)
                        or (epoch == args.n_epochs - 1)):
                    # keys = main_dict.keys()
                    update_dict = main_dict.copy()

                    # copy in the input parameters
                    if split_input:
                        if args.n_layers <= 1 and not split_output:
                            for idx, sub_dict in zip(feats_idx[0], sub_dict_list):
                                update_dict['layers.0.weight'][idx, :] = sub_dict['layers.0.weight']
                        else:
                            for i, sub_dict in enumerate(sub_dict_list):
                                curr_idx = feats_idx[0][i]
                                next_idx = feats_idx[1][i]
                                correct_rows = update_dict['layers.0.weight'][curr_idx, :]
                                correct_rows[:, next_idx] = sub_dict['layers.0.weight']
                                update_dict['layers.0.weight'][curr_idx, :] = correct_rows
                    else:
                        if args.n_layers <= 1 and not split_output:
                            update_dict['layers.0.weight'] = sum(
                                sub_dict['layers.0.weight'] for sub_dict in sub_dict_list) / len(sub_dict_list)
                        else:
                            visited = {}
                            # # Test dict to prove correctness
                            # test_dict = {}
                            # test_dict['layers.0.weight'] = update_dict['layers.0.weight']
                            for i, sub_dict in enumerate(sub_dict_list):
                                next_idx = feats_idx[1][i]
                                for j, k in enumerate(next_idx):
                                    if not k.item() in visited:
                                        visited[k.item()] = sub_dict['layers.0.weight'][:, j]
                                    else:
                                        # print("Input dup.", k.item())
                                        visited[k.item()] = (visited[k.item()] + sub_dict['layers.0.weight'][:, j]) / 2
                                # update_dict['layers.0.weight'][:, next_idx] = sub_dict['layers.0.weight']
                            for k in visited:
                                update_dict['layers.0.weight'][:, k] = visited[k]
                            # print("Input: ", torch.equal(test_dict['layers.0.weight'], update_dict['layers.0.weight']))

                    # copy the rest of the parameters
                    for i in range(1, args.n_layers + 1):
                        if i == args.n_layers:
                            if not split_output:
                                update_dict[f'layers.{i - 1}.bias'] = sum(
                                    sub_dict[f'layers.{i - 1}.bias'] for sub_dict in sub_dict_list) / len(sub_dict_list)
                                update_dict[f'layers.{i}.weight'] = sum(
                                    sub_dict[f'layers.{i}.weight'] for sub_dict in sub_dict_list) / len(sub_dict_list)
                            else:
                                for idx, sub_dict in zip(feats_idx[i], sub_dict_list):
                                    update_dict[f'layers.{i - 1}.bias'][idx] = sub_dict[f'layers.{i - 1}.bias']
                                    update_dict[f'layers.{i}.weight'][idx, :] = sub_dict[f'layers.{i}.weight']
                        else:
                            if i >= args.n_layers - 1 and not split_output:
                                sub_biases = {}
                                sub_weights = {}
                                # test_dict = {}
                                # # Test dict to prove correctness
                                # test_dict[f'layers.{i-1}.bias'] = update_dict[f'layers.{i-1}.bias'].detach().clone()
                                # test_dict[f'layers.{i}.weight'] = update_dict[f'layers.{i}.weight'].detach().clone()
                                for idx, sub_dict in zip(feats_idx[i], sub_dict_list):
                                    # Aggregate the parameters similar to FedAvg
                                    for j, k in enumerate(idx):
                                        if not k.item() in sub_biases:
                                            sub_biases[k.item()] = sub_dict[f'layers.{i - 1}.bias'][j]
                                        else:
                                            sub_biases[k.item()] = (sub_biases[k.item()] + sub_dict[f'layers.{i-1}.bias'][j]) / 2
                                        if not k.item() in sub_weights:
                                            sub_weights[k.item()] = sub_dict[f'layers.{i}.weight'][j, :]
                                        else:
                                            sub_weights[k.item()] = (sub_weights[k.item()] + sub_dict[f'layers.{i}.weight'][j, :]) / 2
                                    # update_dict[f'layers.{i - 1}.bias'][idx] = sub_dict[f'layers.{i - 1}.bias']
                                    # update_dict[f'layers.{i}.weight'][idx, :] = sub_dict[f'layers.{i}.weight']
                                for k in sub_biases:
                                    update_dict[f'layers.{i-1}.bias'][k] = sub_biases[k]
                                    update_dict[f'layers.{i}.weight'][k, :] = sub_weights[k]
                                # print(torch.equal(test_dict[f'layers.{i-1}.bias'], update_dict[f'layers.{i - 1}.bias']))
                                # print(torch.equal(test_dict[f'layers.{i}.weight'], update_dict[f'layers.{i}.weight']))
                            else:
                                sub_biases = {}
                                sub_weights = {}
                                # test_dict = {}
                                # test_dict[f'layers.{i - 1}.bias'] = update_dict[f'layers.{i - 1}.bias'].detach().clone()
                                # test_dict[f'layers.{i}.weight'] = update_dict[f'layers.{i}.weight'].detach().clone()
                                for idx, sub_dict in enumerate(sub_dict_list):
                                    curr_idx = feats_idx[i][idx]
                                    next_idx = feats_idx[i + 1][idx]
                                    for j, k in enumerate(curr_idx):
                                        if not k.item() in sub_biases:
                                            sub_biases[k.item()] = sub_dict[f'layers.{i-1}.bias'][j]
                                        else:
                                            sub_biases[k.item()] = (sub_biases[k.item()] + sub_dict[f'layers.{i-1}.bias'][j]) / 2
                                    for cj, ck in enumerate(curr_idx):
                                        for nj, nk in enumerate(next_idx):
                                            if not (ck, nk) in sub_weights:
                                                sub_weights[(ck, nk)] = sub_dict[f'layers.{i}.weight'][cj, nj]
                                            else:
                                                sub_weights[(ck, nk)] = (sub_weights[(ck, nk)] + sub_dict[f'layers.{i}.weight'][cj, nj]) / 2
                                    # update_dict[f'layers.{i - 1}.bias'][curr_idx] = sub_dict[f'layers.{i - 1}.bias']
                                    # correct_rows = update_dict[f'layers.{i}.weight'][curr_idx, :]
                                    # correct_rows[:, next_idx] = sub_dict[f'layers.{i}.weight']
                                    # update_dict[f'layers.{i}.weight'][curr_idx, :] = correct_rows
                                for k in sub_biases:
                                    update_dict[f'layers.{i-1}.bias'][k] = sub_biases[k]
                                for pr in sub_weights:
                                    update_dict[f'layers.{i}.weight'][pr[0].item()][pr[1].item()] = sub_weights[pr]
                                # print('middle', torch.equal(test_dict[f'layers.{i-1}.bias'], update_dict[f'layers.{i - 1}.bias']))
                                # print('middle', torch.equal(test_dict[f'layers.{i}.weight'], update_dict[f'layers.{i}.weight']))
                    model.load_state_dict(update_dict)

            else:
                raise NotImplementedError('Should train with IST')

            if epoch >= 3:
                dur.append(time.time() - t0)

            acc_train = evaluate(model, [x[0] for x in loaders])
            # acc_val = evaluate(model, features, labels, val_mask)
            acc_test = evaluate(model, [x[1] for x in loaders])
            record.append([acc_train, acc_test])

        all_train_acc = [v[0] for v in record]
        all_test_acc = [v[1] for v in record]
        # all_val_acc = [v[1] for v in record]
        # acc = evaluate(model, [x[1] for x in loaders])
        path = "./imgs/gist_nodeshare/" + args.dataset
        name = f"gist_noniid_nodeshare_{args.dataset}_neuron{args.n_hidden}_layer{args.n_layers}_worker{args.num_subnet}_dropout{args.dropout}_lr{args.lr}"
        draw_acc_curve(record, path, name + f"_V{run}")
        training_spec["runs"].append({
            "acc_train": all_train_acc,
            "acc_test": all_test_acc,
            "best_train_acc": f"{max(all_train_acc):.4f}",
            "final_test_acc": f"{all_test_acc[-1]:.4f}",
            "best_test_acc": f"{max(all_test_acc):.4f}"
        })
        avg_best_train += max(all_train_acc)
        avg_best_test += max(all_test_acc)
        # print(f"Best Train Accuracy: {max(all_train_acc):.4f}")
        # print(f"Final Test Accuracy: {acc:.4f}")
        # print(f"Best Test Accuracy: {max(all_test_acc):.4f}")
    
    avg_best_train /= 3
    avg_best_test /= 3
    training_spec["avg_best_train"] = avg_best_train
    training_spec["avg_best_test"] = avg_best_test
    print(f"Average Best Train Accuracy: {avg_best_train:.4f}")
    print(f"Average Best Test Accuracy: {avg_best_test:.4f}")
    json.dump(training_spec, open(os.path.join(path, name) + "_stats.json", "w"), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="PROTEINS")
    parser.add_argument("--use_ist", type=str, default="True",
                        help="whether use IST training")
    parser.add_argument("--iter_per_site", type=int, default=5)
    parser.add_argument("--num_subnet", type=int, default=10,
                        help="number of sub networks")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--split_output", type=str, default="False")
    parser.add_argument("--split_input", type=str, default="False")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=400,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self_loop", type=str, default='True',
                        help="graph self-loop (default=True)")
    parser.add_argument("--use_layernorm", type=str, default='True',
                        help="Whether use layernorm (default=False)")
    parser.add_argument("--non_iid", type=str, default='True',
                        help="Whether use non-iid data (default=True)")
    args = parser.parse_args()
    main(args)
