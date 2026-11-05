# Análisis de grafos construidos a partir de predicciones del modelo
# Equivalente a analisis_dataset_minimal_v5.py pero usando predictions.csv como aristas
#
# Formato esperado de predictions.csv:
#   text_id, source_id, target_id, source_tokens, target_tokens,
#   true_label, score_support, score_attack, predicted_label
#
# Formato esperado de component_<split>.csv (sin cambios):
#   text_id, component_id, component_text, full_text, label, minimalist_labels

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate

# ── Rutas ─────────────────────────────────────────────────────────────────────
COMPONENT_DIR  = "./dataset_minimal/"          # directorio con component_<split>.csv
PREDICTIONS_CSV = "./data_predictions/predictions.csv"          # archivo de predicciones del modelo
RESULT_PATH    = "./results/predictions_v5/"   # directorio de salida

os.makedirs(RESULT_PATH, exist_ok=True)

# ── Carga de datos ─────────────────────────────────────────────────────────────

# Componentes reales (nodos) — igual que en create_dataset_minimal_v5.py
dfc = pd.concat([
    pd.read_csv(os.path.join(COMPONENT_DIR, "component_train.csv")),
    pd.read_csv(os.path.join(COMPONENT_DIR, "component_test.csv")),
    pd.read_csv(os.path.join(COMPONENT_DIR, "component_val.csv")),
], ignore_index=True)

# Predicciones del modelo (aristas)
df_preds = pd.read_csv(PREDICTIONS_CSV)

# Validar columnas mínimas requeridas
required_cols = {"text_id", "source_id", "target_id", "predicted_label"}
missing = required_cols - set(df_preds.columns)
if missing:
    raise ValueError(
        f"predictions.csv le faltan columnas: {missing}\n"
        f"Columnas encontradas: {list(df_preds.columns)}"
    )

# Solo conservar relaciones predichas (support / attack) — sin "none" ni similares
valid_labels = {"support", "attack"}
df_preds = df_preds[df_preds["predicted_label"].isin(valid_labels)].copy()

print(f"Predicciones cargadas: {len(df_preds)} relaciones válidas (support/attack)")
print(f"Componentes cargados:  {len(dfc)} segmentos\n")

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_nodes(text_id: str) -> pd.DataFrame:
    """Nodos (componentes reales) para un text_id dado."""
    return dfc[dfc["text_id"] == text_id].copy()


def get_predicted_edges(text_id: str) -> pd.DataFrame:
    """Aristas predichas para un text_id dado."""
    return df_preds[df_preds["text_id"] == text_id].copy()


def crear_grafo_predicted(df_nodes: pd.DataFrame, df_edges: pd.DataFrame):
    """
    Construye un DiGraph con:
      - Nodos: componentes reales (minimalist_labels como atributo 'component')
      - Aristas: relaciones predichas por el modelo (predicted_label como 'label')
    La dirección es target → source, igual que en create_dataset_minimal_v5.py.
    """
    G = nx.DiGraph()

    # Agregar nodos
    for _, row in df_nodes.iterrows():
        G.add_node(
            row["component_id"],
            component=row["minimalist_labels"]
        )

    # Agregar aristas predichas
    for _, row in df_edges.iterrows():
        # Solo agrega la arista si ambos extremos existen como nodos
        if row["source_id"] in G.nodes and row["target_id"] in G.nodes:
            G.add_edge(
                row["target_id"],   # misma convención que el script original
                row["source_id"],
                label=row["predicted_label"]
            )

    n_comp     = nx.number_weakly_connected_components(G)
    n_isolates = nx.number_of_isolates(G)
    n_nodes    = G.number_of_nodes()
    n_edges    = G.number_of_edges()
    return G, n_nodes, n_edges, n_comp, n_isolates


def plot_grafo(text_id: str, G: nx.DiGraph):
    """Visualización del grafo — igual que en los scripts originales."""
    color_nodo = {
        "MajorClaim": "#e74c3c",
        "Claim":      "#3498db",
        "Premise":    "#2ecc71",
    }
    node_colors = [
        color_nodo.get(G.nodes[n].get("component", ""), "#95a5a6")
        for n in G.nodes()
    ]
    colores_label = {
        "support": "#27ae60",
        "attack":  "#c0392b",
    }
    edge_colors = [colores_label.get(d["label"], "gray") for _, _, d in G.edges(data=True)]

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=30, iterations=100)
    node_labels = {
        n: f"{n}\n{G.nodes[n].get('component', '')}"
        for n in G.nodes()
    }
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="black",
                            font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           arrows=True, arrowsize=25, arrowstyle="-|>",
                           width=2, connectionstyle="arc3,rad=0.1",
                           min_source_margin=30, min_target_margin=30)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=7, font_color="darkgreen")
    leyenda = [
        Line2D([0], [0], color="green", linewidth=2, label="support"),
        Line2D([0], [0], color="red",   linewidth=2, label="attack"),
        Line2D([0], [0], color="gray",  linewidth=2, label="otros"),
    ]
    plt.legend(handles=leyenda, loc="upper left", fontsize=9)
    plt.title(f"Grafo predicho: {text_id}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ── Construcción de grafos por dataset ─────────────────────────────────────────

# Detectar datasets presentes en las predicciones (ej: AAEC, CDCP, ...)
df_preds["dataset"] = df_preds["text_id"].str.rsplit("_", n=1).str[0]
datasets = df_preds["dataset"].unique()
print(f"Datasets en predicciones: {sorted(datasets)}\n")

# Para cada dataset, construir un DataFrame resumen con un grafo por text_id
graphs_by_dataset = {}

for dataset_name in sorted(datasets):
    text_ids = df_preds[df_preds["dataset"] == dataset_name]["text_id"].unique()
    rows = []
    for text_id in text_ids:
        df_nodes = get_nodes(text_id)
        df_edges = get_predicted_edges(text_id)

        if df_nodes.empty:
            # text_id en predicciones pero sin componentes en component_*.csv
            print(f"  [WARN] Sin nodos para {text_id} — se omite")
            continue

        G, n_nodes, n_edges, n_comp, n_isolates = crear_grafo_predicted(df_nodes, df_edges)
        rows.append({
            "text_id":    text_id,
            "seg":        len(df_nodes),
            "graph":      G,
            "n_nodes":    n_nodes,
            "n_edges":    n_edges,
            "n_comp":     n_comp,
            "n_isolates": n_isolates,
        })

    df_dataset = pd.DataFrame(rows)
    graphs_by_dataset[dataset_name] = df_dataset
    # Guardar pkl por dataset (mismo formato que create_dataset_minimal_v5.py)
    out_pkl = os.path.join(RESULT_PATH, f"{dataset_name}_Predicted_Grafo.pkl")
    df_dataset.to_pickle(out_pkl)
    print(f"  Guardado: {out_pkl}  ({len(df_dataset)} textos)")

# ── Métricas — mismo bloque que analisis_dataset_minimal_v5.py ────────────────

acyclic_col      = "acyclic"
cycles_col       = "cycles"
is_tree_col      = "tree"
is_arborescence  = "arborescence"
nodes_roots_col  = "n_roots"


def get_subgrafos(df: pd.DataFrame) -> dict:
    df_cmp = df[(df["n_comp"] != 1) & (df["n_isolates"] == 0)]
    return df_cmp.groupby("n_comp").size().to_dict()


def get_graph_conexo(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["n_comp"] == 1].copy()


dfs_sumary  = []
dfs_indices = {}

for index, (dataset_name, df) in enumerate(graphs_by_dataset.items()):
    dfs_indices[index] = dataset_name

    isolados   = (df["n_isolates"] != 0).sum()
    subgrafos  = get_subgrafos(df)
    df_conexo  = get_graph_conexo(df)
    conexo     = df_conexo.shape[0]

    # Propiedades sobre grafos conexos
    df_conexo[acyclic_col]     = df_conexo["graph"].apply(nx.is_directed_acyclic_graph)
    df_conexo[cycles_col]      = df_conexo["graph"].apply(lambda G: list(nx.simple_cycles(G)))
    df_conexo[is_tree_col]     = df_conexo["graph"].apply(nx.is_tree)
    df_conexo[is_arborescence] = df_conexo["graph"].apply(nx.is_arborescence)
    df_conexo[nodes_roots_col] = df_conexo["graph"].apply(
        lambda G: sum(1 for _, grado in G.in_degree() if grado == 0)
    )

    aciclico = df_conexo[df_conexo[acyclic_col]].shape[0]
    arvore   = df_conexo[df_conexo[is_tree_col]].shape[0]
    nos_raiz = df_conexo[nodes_roots_col].value_counts().sort_index().to_dict()

    dfs_sumary.append({
        "total_text_arg": len(df),
        "isolados":       isolados,
        "subgrafos":      ", ".join(f"{k}: {v}" for k, v in subgrafos.items()),
        "conexo":         conexo,
        "aciclico":       aciclico,
        "nos_raiz":       ", ".join(f"{k}: {v}" for k, v in nos_raiz.items()),
        "arvore":         arvore,
    })

df_resultados = pd.json_normalize(dfs_sumary)
df_resultados = df_resultados.rename(index=dfs_indices)

out_summary = os.path.join(RESULT_PATH, "predicted_summary.pkl")
df_resultados.to_pickle(out_summary)

print("\n── Resumen de métricas (grafos predichos) ──────────────────────────────")
print(tabulate(df_resultados, headers="keys", tablefmt="rounded_outline", showindex=True))





import pandas as pd
import networkx as nx
from tabulate import tabulate


def calcular_metricas_grafos(graphs_by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Para cada dataset en graphs_by_dataset calcula, grafo a grafo:
        - is_directed_acyclic_graph : el grafo no tiene ciclos dirigidos
        - is_weakly_connected       : el grafo es débilmente conexo
        - is_tree                   : el grafo subyacente no-dirigido es un árbol
        - is_arborescence           : árbol dirigido con exactamente una raíz

    Reporta, por dataset:
        total          : número de grafos
        dag            : cuántos son DAG                (valor absoluto)
        dag_%          : ídem en porcentaje
        weakly_conn    : cuántos son débilmente conexos
        weakly_conn_%  : ídem en porcentaje
        tree           : cuántos son árbol
        tree_%         : ídem en porcentaje
        arborescence   : cuántos son arborescencia
        arborescence_% : ídem en porcentaje

    Parameters
    ----------
    graphs_by_dataset : dict
        Clave   → nombre del dataset (str)
        Valor   → DataFrame con al menos la columna "graph" (nx.DiGraph)

    Returns
    -------
    df_resultados : pd.DataFrame
        Una fila por dataset, índice = nombre del dataset.
    """

    # Las 4 propiedades y la función NetworkX correspondiente
    PROPIEDADES = {
        "dag":          nx.is_directed_acyclic_graph,
        "weakly_conn":  nx.is_weakly_connected,
        "tree":         nx.is_tree,
        "arborescence": nx.is_arborescence,
    }

    filas = []
    indices = []

    for dataset_name, df in graphs_by_dataset.items():
        indices.append(dataset_name)
        total = len(df)
        fila = {"total": total}

        for col, nx_func in PROPIEDADES.items():
            # Aplica la función NetworkX a cada grafo; captura errores puntuales
            resultados = df["graph"].apply(lambda g, f=nx_func: _safe_apply(g, f))
            cuenta = int(resultados.sum())
            pct    = round(cuenta / total * 100, 1) if total > 0 else 0.0
            fila[col]           = cuenta
            fila[f"{col}_%"]    = pct

        filas.append(fila)

    # Columnas en el orden declarado en PROPIEDADES
    col_order = ["total"]
    for col in PROPIEDADES:
        col_order += [col, f"{col}_%"]

    df_resultados = pd.DataFrame(filas, index=indices)[col_order]
    return df_resultados


def _safe_apply(graph: nx.DiGraph, func) -> bool:
    """Llama func(graph) y devuelve False si lanza excepción."""
    try:
        return bool(func(graph))
    except Exception:
        return False


def reporte_metricas_grafos(
    graphs_by_dataset: dict[str, pd.DataFrame],
    titulo: str = "Métricas de grafos",
) -> pd.DataFrame:
    """
    Calcula y imprime las métricas con el mismo estilo tabulate
    usado en analisis_dataset_minimal_v5.py.

    Returns
    -------
    df_resultados : pd.DataFrame  (también impreso en consola)
    """
    df = calcular_metricas_grafos(graphs_by_dataset)

    print(f"\n── {titulo} {'─' * max(0, 60 - len(titulo))}")
    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=True))
    return df


# ── Uso ───────────────────────────────────────────────────────────────────────
# Pega estas dos líneas donde tengas graphs_by_dataset disponible:
#
#   from metricas_grafos import reporte_metricas_grafos
reporte_metricas_grafos(graphs_by_dataset, titulo="Grafos predichos")