import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
from dgl.data import CoraGraphDataset

import networkx as nx
import matplotlib.pyplot as plt

# ── 1. Cargar el dataset ──────────────────────────────────────────
data = CoraGraphDataset()   # descarga automática ~3MB
g    = data[0]              # el grafo completo: 2708 nodos, 5429 aristas

# Extraer features y etiquetas
h      = g.ndata['feat']   # (2708, 1433) — vector de palabras por paper
labels = g.ndata['label']  # (2708,)      — categoría real: 0..6

# Máscaras de train / val / test (vienen con el dataset)
train_mask = g.ndata['train_mask']  # 140 nodos
val_mask   = g.ndata['val_mask']    # 500 nodos
test_mask  = g.ndata['test_mask']   # 1000 nodos

nx_g = train_mask.to_networkx()
nx.draw(nx_g, with_labels=True)
plt.show()

# Añadir self-loops (cada nodo se atiende a sí mismo)
g = dgl.add_self_loop(g)

# ── 2. Definir el modelo ──────────────────────────────────────────
class GAT(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_heads):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden, num_heads=num_heads, feat_drop=0.6, attn_drop=0.6)
        self.conv2 = GATConv(hidden * num_heads, out_feats, num_heads=1, feat_drop=0.6, attn_drop=0.6)

    def forward(self, g, x):
        x = self.conv1(g, x).flatten(1)   # (2708, 8*8) = (2708, 64)
        x = F.elu(x)                       # ELU en vez de ReLU (mejor en Cora)
        x = self.conv2(g, x).squeeze(1)   # (2708, 7)
        return x

# in_feats=1433 palabras → hidden=8 por cabeza × 8 cabezas → 7 clases
model = GAT(in_feats=1433, hidden=8, out_feats=7, num_heads=8)

# ── 3. Entrenamiento ──────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train()
    logits = model(g, h)                          # out: (2708, 7)
    loss   = F.cross_entropy(logits[train_mask],  # solo 140 nodos etiquetados
                              labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            logits  = model(g, h)
            pred    = logits.argmax(dim=1)
            acc_val = (pred[val_mask] == labels[val_mask]).float().mean()
        print(f"Época {epoch:3d}  loss={loss:.4f}  val_acc={acc_val:.4f}")

# ── 4. Evaluación final ───────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits    = model(g, h)
    pred      = logits.argmax(dim=1)        # clase con mayor probabilidad
    test_acc  = (pred[test_mask] == labels[test_mask]).float().mean()

print(f"\nTest accuracy: {test_acc:.4f}")   # → ~0.83