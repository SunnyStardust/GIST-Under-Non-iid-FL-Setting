import os

import dgl
from matplotlib import pyplot as plt
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data, LegacyTUDataset
from gcn import GCN, GraphGCN
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader

os.environ["DGL_REPO"] = "http://data.dgl.ai/"



def draw_acc_curve(record, save_root, save_filename):
    acc_train = np.array(record)[:, 0]
    acc_test = np.array(record)[:, 1]
    plt.title("Validation Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(len(acc_test)), acc_train, 'red', label="train")
    plt.plot(range(len(acc_test)), acc_test, 'orange', label="test")
    plt.legend()
    np.save(os.path.join(save_root, 'record_' + save_filename), record)
    plt.savefig(os.path.join(save_root, save_filename + '.jpg'))
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
    assert args.self_loop in ['True', 'False'], ["Only True or False for self_loop, get ",
                                                 args.self_loop]
    assert args.use_layernorm in ['True', 'False'], ["Only True or False for use_layernorm, get ",
                                                     args.use_layernorm]
    self_loop = (args.self_loop == 'True')
    use_layernorm = (args.use_layernorm == 'True')
    global t0
    data = LegacyTUDataset(args.dataset)

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
    print(len(train_dataloader), len(test_dataloader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create GCN model
    model = GraphGCN(
        len(data[0][0].ndata['feat'][0]), args.n_hidden, data.num_labels, args.n_layers, F.relu,
        args.dropout, use_layernorm)
    model = model.to(device)
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    record = []
    dur = []
    for epoch in range(args.n_epochs):
        if args.lr_scheduler:
            if epoch == int(0.5 * args.n_epochs):
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] / 10
            elif epoch == int(0.75 * args.n_epochs):
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] / 10
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        epoch_loss = 0
        for batched_graph, labels in train_dataloader:
            bnn = batched_graph.batch_num_nodes()
            bne = batched_graph.batch_num_edges()
            batched_graph = dgl.remove_self_loop(batched_graph)
            batched_graph = dgl.add_self_loop(batched_graph)
            batched_graph.set_batch_num_edges(bne)
            batched_graph.set_batch_num_nodes(bnn)
            labels = labels.long()
            # labels = torch.tensor([x[0] for x in labels]).long()
            pred = model(batched_graph, batched_graph.ndata['feat'].float())
            loss = loss_fcn(pred, labels)
            epoch_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'{epoch}:', epoch_loss / len(train_dataloader))

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
    draw_acc_curve(record, "./", f"gcn_graph_{args.dataset}_acc")
    print(f"Best Train Accuracy: {max(all_train_acc):.4f}")
    print(f"Final Test Accuracy: {acc:.4f}")
    # print(f"Best Val Accuracy: {max(all_val_acc):.4f}")
    print(f"Best Test Accuracy: {max(all_test_acc):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="DD")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=.001,
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
    parser.add_argument("--lr_scheduler", action='store_true', default=False,
                        help="Use LR scheduler")
    parser.add_argument("--use_layernorm", type=str, default='True',
                        help="Whether use layernorm (default=False)")
    args = parser.parse_args()
    main(args)
