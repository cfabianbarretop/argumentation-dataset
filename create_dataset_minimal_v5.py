# Primer paso: crear los dataset se base a datos .csv de:
# (CDCP, AAEC, ArGPT, MTC, AASD)

import pandas as pd
from tabulate import tabulate
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

result_path = "./results/minimal_v5/"
path_minimal = "./dataset_minimal/"

columna = "text_id"
name_graph_minimal = "_Minimal_Grafo.pkl"

# Cargar 0 CSV
# Componente
dfv_train = pd.read_csv(path_minimal + "component_train.csv")
dfv_test = pd.read_csv(path_minimal + "component_test.csv")
dfv_value = pd.read_csv(path_minimal + "component_val.csv")
# Relações
dfr_train = pd.read_csv(path_minimal + "relation_train.csv")
dfr_test = pd.read_csv(path_minimal + "relation_test.csv")
dfr_val = pd.read_csv(path_minimal + "relation_val.csv")

# Misturamos todo
dfc = pd.concat(
    [dfv_train, dfv_test, dfv_value],
    ignore_index=True
)
dfr = pd.concat(
    [dfr_train, dfr_test, dfr_val],
    ignore_index=True
)

# Faz um conteo de registros de cada dataset
conteo = (
    dfc[columna]
    .drop_duplicates()
    .str.rsplit("_", n=1).str[0]
    .value_counts()
    .reset_index()
)

conteo.columns = ["valor", "cantidad"]

conteo["porcentaje"] = (
    conteo["cantidad"] / len(dfc[columna].drop_duplicates()) * 100
).round(2)

print(tabulate(conteo.reset_index(), headers='keys', tablefmt='rounded_outline', showindex=False))
total = conteo["cantidad"].sum()
print("Total:", total)

def get_dfc_dataset_sumary(indice_tableta_dataset):
    name_dataset = conteo["valor"][indice_tableta_dataset]

    # Só ids dos componentes
    dfc_name_dataset = dfc[dfc[columna].str.contains(name_dataset, na=False)].copy()
    dfc_name_dataset_summary = (
        dfc_name_dataset.groupby(columna)
        .size()
        .reset_index(name="seg")
    )
    return dfc_name_dataset_summary

# Obtener o nó por text_id
def get_nodes(text_id):
    dfc_filter = dfc[dfc[columna].str.contains(text_id, na=False)].copy()
    return dfc_filter
# Obtener as relações por text_id
def get_edges(text_id):
    dfr_filter = dfr[dfr[columna] == text_id]
    dfr_filter = dfr_filter.dropna()
    return dfr_filter

# Obter dados para crear um grafo
def get_data_graph(dfc_component, dfr_relation):

    dfc_component = dfc_component.rename(columns={
        "label": "labels_component",
        "minimalist_labels": "minimalist_labels_component"
    })

    dfr_relation = dfr_relation.rename(columns={
        "label": "labels_relation",
    })
    
    dfr_relation = dfr_relation.merge(
        dfc_component[['component_id', 'labels_component', 'minimalist_labels_component']],
        left_on='source_id',
        right_on='component_id',
        how='left'
    )

    dfr_relation = dfr_relation.rename(columns={
        "labels_component": "label_component_source",
        "minimalist_labels_component": "minimal_label_component_source",

    })

    dfr_relation = dfr_relation.merge(
        dfc_component[['component_id', 'labels_component', 'minimalist_labels_component']],
        left_on='target_id',
        right_on='component_id',
        how='left'
    )

    dfr_relation = dfr_relation.rename(columns={
        "labels_component": "label_component_target",
        "minimalist_labels_component": "minimal_label_component_target",

    })

    # Seleccionar solo las columnas que necesitas
    df_merged = dfr_relation[[
        "minimalist_labels",
        "labels_relation",
        "source_id",
        "label_component_source",
        "minimal_label_component_source",
        "target_id",
        "label_component_target",
        "minimal_label_component_target"
    ]]

    return df_merged

# Creação do grafo
def crear_grafo(df_sub):  
    # Crear grafo dirigido
    G = nx.DiGraph()
    
    for _, row in df_sub.iterrows():
        # G.add_edge(row["source_id"], row["target_id"], label=row["minimalist_labels"])
        G.add_edge(row["target_id"], row["source_id"], label=row["minimalist_labels"])
        G.nodes[row["source_id"]]["component"] = row["minimal_label_component_source"]
        G.nodes[row["target_id"]]["component"] = row["minimal_label_component_target"]

    n_comp = nx.number_weakly_connected_components(G)
    n_isolates = nx.number_of_isolates(G)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    return G, n_nodes, n_edges, n_comp, n_isolates

# Disenho do grafo
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

# ------------------------------------------------------------------
# Processo
# ------------------------------------------------------------------
for index, row in tqdm(conteo.iterrows(), total=len(conteo)):
    name_graph = row['valor'] + name_graph_minimal
    dfc_name_dataset_summary = get_dfc_dataset_sumary(indice_tableta_dataset=index)
    info = "info"
    g = "graph"
    n_node = "n_nodes"
    n_edge = "n_edges"
    n_comp = "n_comp"
    n_isol = "n_isolates"

    dfc_name_dataset_summary[info] = None
    dfc_name_dataset_summary[g] = None
    dfc_name_dataset_summary[n_node] = None
    dfc_name_dataset_summary[n_edge] = None
    dfc_name_dataset_summary[n_comp] = None
    dfc_name_dataset_summary[n_isol] = None


    for idx, component_id in dfc_name_dataset_summary[columna].items():
        df_nodes = get_nodes(component_id)
        df_edges = get_edges(component_id)
        df_graph = get_data_graph(df_nodes, df_edges)
        graph, n_nodes, n_edges, n_compt, n_isolates = crear_grafo(df_graph)
        dfc_name_dataset_summary.at[idx, info] = df_graph
        dfc_name_dataset_summary.at[idx, g] = graph
        dfc_name_dataset_summary.at[idx, n_node] = n_nodes
        dfc_name_dataset_summary.at[idx, n_edge] = n_edges
        dfc_name_dataset_summary.at[idx, n_comp] = n_compt
        dfc_name_dataset_summary.at[idx, n_isol] = n_isolates
    dfc_name_dataset_summary.to_pickle(result_path + name_graph)

