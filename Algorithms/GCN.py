import json
import torch
import random
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import pickle
import os

import warnings
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


idx_features_labels = np.genfromtxt("./data/content.csv", dtype=np.dtype(str))
edges_unordered = np.genfromtxt("./data/25_link.csv", dtype=np.int32)

idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
labels = idx_features_labels[:, -1]

classes_dict = {'0': 0, '1': 1}
labels = np.array(list(map(classes_dict.get, labels)))

idx_dict = {j: i for i, j in enumerate(idx)}
edges = np.array(list(map(idx_dict.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

features = torch.FloatTensor(np.array(normalize(features).todense())).to(DEVICE)
labels = torch.LongTensor(labels).to(DEVICE)


adj_edges = [[],[]]
for i in edges:
    adj_edges[0].append(i[0])
    adj_edges[1].append(i[1])
adj_edges = torch.LongTensor(adj_edges).to(DEVICE)


# 检查随机文件是否存在，存在则读取，不存在则生成并存储
mask = []
if os.path.exists('mask'):
    f = open('mask', 'rb')
    mask = pickle.load(f)
    f.close()
else:
    for i in range(0, 4508):
        random_mask = random.randint(1, 10)
        mask.append(random_mask)
    f = open('mask', 'wb')
    pickle.dump(mask, f)
    f.close()

train_mask = []
val_mask = []
test_mask = []

for i in mask:
    if i <= 7:
        train_mask.append(True)
        val_mask.append(False)
        test_mask.append(False)
    elif i == 8:
        train_mask.append(False)
        val_mask.append(True)
        test_mask.append(False)
    else:
        train_mask.append(False)
        val_mask.append(False)
        test_mask.append(True)

train_mask = torch.tensor(train_mask).to(DEVICE)
val_mask = torch.tensor(val_mask).to(DEVICE)
test_mask = torch.tensor(test_mask).to(DEVICE)


from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

for cishu in range (0,50):

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(40, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, 2)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            # x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    model = GCN(hidden_channels=512).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # 优化器


    def train():
        model.train()
        x, edge_index = features, adj_edges
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        return loss

    def test():
        model.eval()
        out = model(features, adj_edges)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        result_pred = pred[test_mask].cpu().numpy()
        result_true = labels[test_mask].cpu().numpy()

        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for i in range(0, len(result_pred)):
            if result_pred[i] == 1 and result_true[i] == 1:
                tp = tp + 1
            elif result_pred[i] == 0 and result_true[i] == 1:
                fn = fn + 1
            elif result_pred[i] == 1 and result_true[i] == 0:
                fp = fp + 1
            elif result_pred[i] == 0 and result_true[i] == 0:
                tn = tn + 1

        from sklearn.metrics import precision_score
        precision_scor = precision_score(result_true, result_pred)

        from sklearn.metrics import recall_score
        recall_scor = recall_score(result_true, result_pred)

        from sklearn.metrics import roc_auc_score
        auc_scor = roc_auc_score(result_true, result_pred)

        return tp, fn, fp, tn, precision_scor, recall_scor, auc_scor

    best_precision_scor = best_recall_scor = best_auc_scor = best_balanced_accurar = 0
    best_tp = best_fn = best_fp = best_tn = 0

    for epoch in range(1, 1000):
        train_loss = train()
        tp, fn, fp, tn, precision_scor, recall_scor, auc_scor = test()
        # 以auc为准
        if auc_scor > best_auc_scor:
            best_tp = tp
            best_fn = fn
            best_fp = fp
            best_tn = tn
            best_recall_scor = recall_scor
            best_precision_scor = precision_scor
            best_auc_scor = auc_scor

    log = 'bufen: {:03d}, best_tp: {:d}, best_fn: {:d}, best_fp: {:d}, best_tn: {:d}, best_recall_score: {:.4f}, best_precision_score: {:.4f}, best_auc_score: {:.4f}'
    print(log.format(cishu, best_tp, best_fn, best_fp, best_tn, best_recall_scor, best_precision_scor,best_auc_scor,best_balanced_accurar))