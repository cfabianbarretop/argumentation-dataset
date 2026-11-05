# Segundo paso: anális os dataset baseado em.csv de:
# (CDCP, AAEC, ArGPT, MTC, AASS)

import pandas as pd
from tabulate import tabulate
import networkx as nx
import matplotlib.pyplot as plt

result_path = "./results/minimal_v5/"
name_dataframe_summary = "dataframe_summary.pkl"

# Desenhar o grafo
def plot_grafo(text_id, G):
    # Cor por nó
    color_nodo = {
        "MajorClaim" : "#e74c3c",   # Vermelho
        "Claim"      : "#3498db",   # Azul
        "Premise"    : "#2ecc71",   # Verde
    }
    node_colors = [
        color_nodo.get(G.nodes[n].get("component", ""), "#95a5a6")
        for n in G.nodes()
    ]

    colores_label = {
        "support" : "#27ae60",   # Verde oscuro
        "attack"  : "#c0392b",   # Vermelho oscuro
    }
    edge_colors   = [colores_label.get(d["label"], "gray") for _, _, d in G.edges(data=True)]
    
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=30, iterations=100)
    
    # Etiquetas de nodos con ID + tipo de componente
    node_labels = {
        n: f"{n}\n{G.nodes[n].get('component', '')}"
        for n in G.nodes()
    }

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="black", font_size=8, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        arrows=True, arrowsize=25, arrowstyle="-|>",
        width=2, connectionstyle="arc3,rad=0.1",
        min_source_margin=30, min_target_margin=30,
    )
    
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color="darkgreen")
    
    from matplotlib.lines import Line2D
    leyenda = [
        Line2D([0], [0], color="green", linewidth=2, label="support"),
        Line2D([0], [0], color="red",   linewidth=2, label="attack"),
        Line2D([0], [0], color="gray",  linewidth=2, label="otros"),
    ]
    plt.legend(handles=leyenda, loc="upper left", fontsize=9)
    plt.title(f"Grafo argumentativo: {text_id}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Cargar el DataFrame
df_cdcp = pd.read_pickle(result_path + "CDCP_Minimal_Grafo.pkl")
df_aaec = pd.read_pickle(result_path + "AAEC_Minimal_Grafo.pkl")
df_argpt = pd.read_pickle(result_path + "ArGPT_Minimal_Grafo.pkl")
df_mtc = pd.read_pickle(result_path + "MTC_Minimal_Grafo.pkl")
df_aasd = pd.read_pickle(result_path + "AASD_Minimal_Grafo.pkl")

df_AbstRCT_glaucoma = pd.read_pickle(result_path + "AbstRCT_glaucoma_Minimal_Grafo.pkl")
df_AbstRCT_mixed = pd.read_pickle(result_path + "AbstRCT_mixed_Minimal_Grafo.pkl")
df_AbstRCT_neoplasm = pd.read_pickle(result_path + "AbstRCT_neoplasm_Minimal_Grafo.pkl")

# lista de Dataframes
dfs = {
    "CDCP":df_cdcp,
    "AAEC":df_aaec,
    "ArGPT":df_argpt,
    "MTC":df_mtc,
    "AASD":df_aasd,
    "Glaucoma":df_AbstRCT_glaucoma,
    "Mixed":df_AbstRCT_mixed,
    "Neoplasm":df_AbstRCT_neoplasm,
}

# --------------------------------------------------------------------------------
# Processo
# --------------------------------------------------------------------------------
total_text_arg = 0
isolados = 0
subgrafos = {}
conexo = 0
aciclico = 0
arvore = 0
nos_raiz = {}
dfs_sumary = []
dfs_indices = {}

def get_subgrafos(df):
    df_comparacion = df[
        (df["n_comp"] != 1) &
        (df["n_isolates"] == 0)
    ]
    conteo = df_comparacion.groupby("n_comp").size()
    return conteo.to_dict()

def get_graph_conexo(df):
    df_comparacion = df[
        (df["n_comp"] == 1)
    ]
    return df_comparacion

# Caracteristicas do grafo
acyclic = "acyclic"
cycles = "cycles"
is_tree = "tree"
is_arborescence = "arborescence"
nodes_roots = "n_roots"

for index, (key, df) in enumerate(dfs.items()):
    dfs_indices[index] = key
    total_text_arg  = len(df)
    isolados = ((df["n_isolates"] != 0)).sum()
    subgrafos = get_subgrafos(df=df)
    df_conexo = get_graph_conexo(df)
    conexo = df_conexo.shape[0]
    # Aciclico e ciclos
    df_conexo[acyclic] = df_conexo["graph"].apply(lambda graph: nx.is_directed_acyclic_graph(graph))
    df_conexo[cycles] = df_conexo["graph"].apply(lambda graph: list(nx.simple_cycles(graph)))
    # Arbol
    df_conexo[is_tree] = df_conexo["graph"].apply(lambda graph: nx.is_tree(graph))
    df_conexo[is_arborescence] = df_conexo["graph"].apply(lambda graph: nx.is_arborescence(graph))
    #Numero de nodos raiz
    df_conexo[nodes_roots] = df_conexo["graph"].apply(lambda G: sum(1 for _, grado in G.in_degree() if grado == 0))
    
    aciclico = df_conexo[df_conexo[acyclic].apply(lambda x: x == True)].shape[0]
    arvore = df_conexo[df_conexo[is_tree].apply(lambda x: x == True)].shape[0]
    nos_raiz = df_conexo[nodes_roots].value_counts().sort_index().to_dict()
    dfs_sumary.append({
        "total_text_arg":total_text_arg,
         "isolados":isolados,
         "subgrafos": ", ".join(f"{k}: {v}" for k, v in subgrafos.items()),
         "conexo":conexo,
         "aciclico":aciclico,
         "nos_raiz": ", ".join(f"{k}: {v}" for k, v in nos_raiz.items()),
         "arvore":arvore,
        }
    )


df_resultados = pd.json_normalize(dfs_sumary)
df_resultados = df_resultados.rename(index=dfs_indices)
# df_resultados.to_pickle(result_path + name_dataframe_summary)
print(tabulate(df_resultados, headers='keys', tablefmt='rounded_outline', showindex=True))