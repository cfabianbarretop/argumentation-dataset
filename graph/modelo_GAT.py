import dgl
import torch
import torch.nn as nn
from dgl.nn import GATConv
import networkx as nx
import matplotlib.pyplot as plt

class GAT(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_heads):
        super().__init__()
        # Capa 1: multi-head attention
        self.conv1 = GATConv(in_feats, hidden, num_heads=num_heads)
        # Capa 2: una sola cabeza para la salida
        self.conv2 = GATConv(hidden * num_heads, out_feats, num_heads=1)

    def forward(self, g, x):
        # Capa 1: concatenar las cabezas → (N, num_heads * hidden)
        x = self.conv1(g, x)
        x = x.flatten(1)        # (N, num_heads, hidden) → (N, num_heads*hidden)
        x = torch.relu(x)

        # Capa 2: una cabeza → (N, 1, out_feats)
        x = self.conv2(g, x)
        x = x.squeeze(1)        # (N, out_feats)
        return x

# Uso
g = dgl.graph(([0,1,2,3,4], [1,2,3,4,0]))
g = dgl.add_self_loop(g)

# nx_g = g.to_networkx()
# nx.draw(nx_g, with_labels=True)
# plt.show()

model = GAT(in_feats=16, hidden=8, out_feats=4, num_heads=4)
h = torch.randn(5, 16) # h = (Nodo, Caracteristicas)
out = model(g, h)  # (4, 4)
# out[0] → representación final del nodo 0
# out[1] → representación final del nodo 1
# out[2] → representación final del nodo 2
# out[3] → representación final del nodo 3

# out_feats = número de clases
pred = out.argmax(dim=1)   # clase predicha por nodo
print(pred)

graph_repr = out.mean(dim=0)   # promedio de todos los nodos
print(graph_repr)

score = (out[0] * out[4]).sum()   # similitud entre nodo 0 y nodo 3
print(score)
