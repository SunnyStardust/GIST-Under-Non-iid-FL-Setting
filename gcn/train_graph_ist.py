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

def draw_acc_curve(record, save_root, save_filename):
    acc_train = np.array(record)[:, 0]
    acc_test = np.array(record)[:, 1]
    plt.title("Validation Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(len(acc_test)), acc_train, 'red', label="train")
    plt.plot(range(len(acc_test)), acc_test, 'orange', label="test")
    plt.legend()
    np.save(os.path.join(save_root, 'record_'+save_filename), record)
    plt.savefig(os.path.join(save_root, save_filename+'.jpg'))
    plt.close()


def evaluate(model, loader):
    model.eval()
    a = 0
    b = 0
    with torch.no_grad():
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
    assert args.use_random_proj in ['True', 'False'], ["Only True or False for use_random_proj, get ",
                                                       args.use_random_proj]
    use_ist = (args.use_ist == 'True')
    split_input = (args.split_input == 'True')
    split_output = (args.split_output == 'True')
    self_loop = (args.self_loop == 'True')
    use_layernorm = (args.use_layernorm == 'True')
    use_random_proj = (args.use_random_proj == 'True')

    # make sure hidden layer is the correct shape
    assert (args.n_hidden % args.num_subnet) == 0
    global t0
    data = LegacyTUDataset(args.dataset)
    in_feats = len(data[0][0].ndata['feat'][0])

    import random
    graphs = [x for x in data]
    random.shuffle(graphs)

    num_examples = len(data)
    num_train = int(num_examples * 0.8)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))
    train_dataloader = GraphDataLoader(
        graphs, sampler=train_sampler, batch_size=5, drop_last=False)
    test_dataloader = GraphDataLoader(
        graphs, sampler=test_sampler, batch_size=5, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create GCN model
    model = GraphGCN(
        len(data[0][0].ndata['feat'][0]), args.n_hidden, data.num_labels, args.n_layers, F.relu,
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
                    feats_idx.append(torch.chunk(torch.randperm(in_feats), num_subnet))
                else:
                    feats_idx.append(None)

                # create hidden layer partitions
                for i in range(1, args.n_layers):
                    feats_idx.append(torch.chunk(torch.randperm(args.n_hidden), num_subnet))

                # create output layer partitions
                if split_output:
                    feats_idx.append(torch.chunk(torch.randperm(args.n_hidden), num_subnet))
                else:
                    feats_idx.append(None)

            for subnet_id in range(args.num_subnet):
                if (epoch % args.iter_per_site) == 0.:
                    # create the sub model to train
                    sub_model = GraphGCN(
                        in_feats, args.n_hidden, data.num_labels,
                        args.n_layers, F.relu, args.dropout, use_layernorm,
                        split_input, split_output, args.num_subnet)
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
                for batched_graph, labels in train_dataloader:
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
                print(f'{epoch}:', epoch_loss / len(train_dataloader))

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
                        for i, sub_dict in enumerate(sub_dict_list):
                            next_idx = feats_idx[1][i]
                            update_dict['layers.0.weight'][:, next_idx] = sub_dict['layers.0.weight']

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
                            for idx, sub_dict in zip(feats_idx[i], sub_dict_list):
                                update_dict[f'layers.{i - 1}.bias'][idx] = sub_dict[f'layers.{i - 1}.bias']
                                update_dict[f'layers.{i}.weight'][idx, :] = sub_dict[f'layers.{i}.weight']
                        else:
                            for idx, sub_dict in enumerate(sub_dict_list):
                                curr_idx = feats_idx[i][idx]
                                next_idx = feats_idx[i + 1][idx]
                                update_dict[f'layers.{i - 1}.bias'][curr_idx] = sub_dict[f'layers.{i - 1}.bias']
                                correct_rows = update_dict[f'layers.{i}.weight'][curr_idx, :]
                                correct_rows[:, next_idx] = sub_dict[f'layers.{i}.weight']
                                update_dict[f'layers.{i}.weight'][curr_idx, :] = correct_rows
                model.load_state_dict(update_dict)

        else:
            raise NotImplementedError('Should train with IST')

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc_train = evaluate(model, train_dataloader)
        # acc_val = evaluate(model, features, labels, val_mask)
        acc_test = evaluate(model, test_dataloader)
        record.append([acc_train, acc_test])

    all_train_acc = [v[0] for v in record]
    all_test_acc = [v[1] for v in record]
    # all_val_acc = [v[1] for v in record]
    acc = evaluate(model, test_dataloader)
    draw_acc_curve(record, "./", f"gist_graph_{args.dataset}_acc")
    print(f"Best Train Accuracy: {max(all_train_acc):.4f}")
    print(f"Final Test Accuracy: {acc:.4f}")
    # print(f"Best Val Accuracy: {max(all_val_acc):.4f}")
    print(f"Best Test Accuracy: {max(all_test_acc):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="DD")
    parser.add_argument("--use_ist", type=str, default="True",
                        help="whether use IST training")
    parser.add_argument("--iter_per_site", type=int, default=5)
    parser.add_argument("--num_subnet", type=int, default=2,
                        help="number of sub networks")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--split_output", type=str, default="False")
    parser.add_argument("--split_input", type=str, default="False")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self_loop", type=str, default='True',
                        help="graph self-loop (default=True)")
    parser.add_argument("--use_layernorm", type=str, default='True',
                        help="Whether use layernorm (default=False)")
    parser.add_argument("--use_random_proj", type=str, default='True',
                        help="Whether use random projection to densitify (default=False)")
    args = parser.parse_args()
    main(args)
