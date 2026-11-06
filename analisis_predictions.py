# Analysis of graphs constructed from model predictions
# Equivalent to analisis_dataset_minimal_v5.py, but using predictions.csv to define the edges
#
# Expected format for predictions.csv:
#   text_id, source_id, target_id, source_tokens, target_tokens,
#   true_label, score_support, score_attack, predicted_label
#
# Expected format for component_<split>.csv (unchanged):
#   text_id, component_id, component_text, full_text, label, minimalist_labels

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tabulate import tabulate

import pickle

# ── Paths ─────────────────────────────────────────────────────────────────────
COMPONENT_DIR  = "./dataset_minimal/"                       # path to component_<split>.csv
PREDICTIONS_CSV = "./data_predictions/predictions.csv"      # path to predictions.csv
RESULT_PATH    = "./results/predictions_v5/"                # path to output directory

os.makedirs(RESULT_PATH, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────

# True components (nodes) — same as in create_dataset_minimal_v5.py
dfc = pd.concat([
    pd.read_csv(os.path.join(COMPONENT_DIR, "component_train.csv")),
    pd.read_csv(os.path.join(COMPONENT_DIR, "component_test.csv")),
    pd.read_csv(os.path.join(COMPONENT_DIR, "component_val.csv")),
], ignore_index=True)

# Predictions of the model (edges)
df_preds = pd.read_csv(PREDICTIONS_CSV)

# Validate required columns
required_cols = {"text_id", "source_id", "target_id", "predicted_label"}
missing = required_cols - set(df_preds.columns)
if missing:
    raise ValueError(
        f"predictions.csv missing columns: {missing}\n"
        f"Columns found: {list(df_preds.columns)}"
    )

valid_labels = {"support", "attack"}
df_preds = df_preds[df_preds["predicted_label"].isin(valid_labels)].copy()

print(f"Predictions loaded: {len(df_preds)} valid relations (support/attack)")
print(f"Invalid predictions filtered out: {len(df_preds) - len(df_preds[df_preds['predicted_label'].isin(valid_labels)])}")
print(f"True components loaded:  {len(dfc)} segments\n")

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_nodes(text_id: str) -> pd.DataFrame:
    """True Nodes for a given text_id."""
    return dfc[dfc["text_id"] == text_id].copy()


def get_predicted_edges(text_id: str) -> pd.DataFrame:
    """Predicted edges for a given text_id."""
    return df_preds[df_preds["text_id"] == text_id].copy()


def create_predicted_graph(df_nodes: pd.DataFrame, df_edges: pd.DataFrame):
    """
    Build a directed graph from nodes and predicted edges.
        - Nodes: real components (minimalist_labels as 'component' attribute)
        - Edges: relations predicted by the model (predicted_label as 'label')
    Direction is target → source, same as in create_dataset_minimal_v5.py.
    """
    G = nx.DiGraph()

    # Add nodes
    for _, row in df_nodes.iterrows():
        G.add_node(
            row["component_id"],
            component=row["minimalist_labels"]
        )

    # Add predicted edges
    for _, row in df_edges.iterrows():
        # Only add the edge if both endpoints exist as nodes
        if row["source_id"] in G.nodes and row["target_id"] in G.nodes:
            G.add_edge(
                row["target_id"],   # same convention as the original script
                row["source_id"],
                label=row["predicted_label"]
            )

    n_comp     = nx.number_weakly_connected_components(G)
    n_isolates = nx.number_of_isolates(G)
    n_nodes    = G.number_of_nodes()
    n_edges    = G.number_of_edges()
    return G, n_nodes, n_edges, n_comp, n_isolates


def plot_graph(text_id: str, G: nx.DiGraph):
    color_node = {
        "MajorClaim": "#e74c3c",
        "Claim":      "#3498db",
        "Premise":    "#2ecc71",
    }
    colors_label = {
            "support": "#27ae60",
            "attack":  "#c0392b",
    }
    
    node_colors = [
        color_node.get(G.nodes[n].get("component", ""), "#95a5a6") for n in G.nodes()
    ]
    
    edge_colors = [
        colors_label.get(d["label"], "gray") for _, _, d in G.edges(data=True)
    ]

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
    plt.title(f"Predicted graph: {text_id}", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ── Build graphs per dataset ─────────────────────────────────────────

# Datasets in predictions (AAEC, CDCP, ...)
df_preds["dataset"] = df_preds["text_id"].str.rsplit("_", n=1).str[0]
datasets = df_preds["dataset"].unique()
print(f"Datasets in predictions: {sorted(datasets)}\n")


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

        G, n_nodes, n_edges, n_comp, n_isolates = create_predicted_graph(df_nodes, df_edges)
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


def calc_graph_metrics(graphs_by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each dataset in graphs_by_dataset, calculate the following for each graph:
        - is_directed_acyclic_graph:    the graph has no directed cycles
        - is_weakly_connected:          the graph is weakly connected
        - is_tree:                      the underlying undirected graph is a tree
        - is_arborescence:              a directed tree with exactly one root

    Report, by dataset:
        - total          : number of graphs
        - dag            : number of DAGs                 (absolute value)
        - dag_%          : same, as a percentage
        - weakly_conn    : number of weakly connected graphs
        - weakly_conn_%  : same, as a percentage
        - tree           : number of trees
        - tree_%         : same, as a percentage
        - arborescence   : number of arborescences
        - arborescence_% : same, as a percentage

    Parameters
    ----------
    graphs_by_dataset : dict
        - Key   → dataset name (str)
        - Value → DataFrame containing at least the "graph" column (nx.DiGraph)

    Returns
    -------
    df_results : pd.DataFrame
        - One row per dataset, index = dataset name.
    """

    PROPERTIES = {
        "isolates":     nx.number_of_isolates,
        "subgraphs":    nx.number_connected_components,
        "weakly_conn":  nx.is_weakly_connected,
        "dag":          nx.is_directed_acyclic_graph,
        "tree":         nx.is_tree,
        "arborescence": nx.is_arborescence,
    }

    results_dict = {}

    for dataset_name, df in graphs_by_dataset.items():
        row = {"total": len(df)}
        
        for col, nx_func in PROPERTIES.items():
            results = df["graph"].apply(lambda g, f=nx_func: _safe_apply(g, f))
            row[col] = int(results.sum())
            
        n_roots = df["graph"].apply(lambda G: sum(1 for _, grado in G.in_degree() if grado == 0))
        dist = n_roots.value_counts().sort_index()
        row["roots"] = ", ".join(f"{k}: {v}" for k, v in dist.items())
        
        results_dict[dataset_name] = row

        print(f"  Trees in {dataset_name}: {get_tree(df)}")

    col_order = ["total"] + list(PROPERTIES.keys()) + ["roots"]
    df_results = pd.DataFrame.from_dict(results_dict, orient="index")[col_order]

    return df_results

def get_tree(df: pd.DataFrame):
    tree_list = df[df["graph"].apply(lambda g: _safe_apply(g, nx.is_tree))]["text_id"].tolist()
    return tree_list


def _safe_apply(graph: nx.DiGraph, func) -> bool:
    """Calls func(graph) and returns False if it raises an exception."""
    try:
        return bool(func(graph))
    except Exception:
        return False


def metrics_report(
    graphs_by_dataset: dict[str, pd.DataFrame],
    title: str = "Graph Metrics Report",
) -> pd.DataFrame:
    """
    Calculate and print the metrics using the same `tabulate` style
    employed in `analisis_dataset_minimal_v5.py`. 

    Returns
    -------
    df_results : pd.DataFrame (also printed to the console)
    """
    df = calc_graph_metrics(graphs_by_dataset)

    print(f"\n── {title} {'─' * max(0, 60 - len(title))}")
    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=True))
    return df


# ── Comparing Ground-truth and Predicted graphs ─────────────────────────────────────────
def compare_graphs(G_true: nx.DiGraph, G_pred: nx.DiGraph) -> dict:
    """
    Compares two graphs edge by edge (including the label support/attack). 
    Returns counts of TP, FP, and FN, as well as the specific edges that differ.
    """
    # Represent each edge as (u, v, label) for comparison with an attribute
    def edge_set(G):
        return {(u, v, d.get("label", "")) for u, v, d in G.edges(data=True)}

    true_edges = edge_set(G_true)
    pred_edges = edge_set(G_pred)

    tp = true_edges & pred_edges
    fp = pred_edges - true_edges
    fn = true_edges - pred_edges

    return {
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "tp_edges": tp, "fp_edges": fp, "fn_edges": fn,
    }


def report_comparison(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    dataset_name: str = "",
) -> pd.DataFrame:
    """
    Compares graphs edge by edge using text_id as the key.
    Reports for each text: TP, FP, FN, Precision, Recall, F1.
    At the end, prints the aggregate for the dataset.
    """
    # Index by text_id to facilitate easy cross-reference
    true_idx = df_true.set_index("text_id")["graph"].to_dict()
    pred_idx = df_pred.set_index("text_id")["graph"].to_dict()

    text_ids = sorted(set(true_idx) & set(pred_idx))
    only_true = set(true_idx) - set(pred_idx)
    only_pred = set(pred_idx) - set(true_idx)

    if only_true:
        print(f"  [WARN] {len(only_true)} text_id only in ground-truth: {sorted(only_true)[:5]}")
    if only_pred:
        print(f"  [WARN] {len(only_pred)} text_id only in predictions: {sorted(only_pred)[:5]}")

    rows = []
    for tid in text_ids:
        r = compare_graphs(true_idx[tid], pred_idx[tid])
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        rows.append({
            "text_id":   tid,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "fp_edges":  r["fp_edges"],
            "fn_edges":  r["fn_edges"],
        })

    df_result = pd.DataFrame(rows)

    # ── Aggregate Summary (micro-average for all edges) ──────────────
    total_tp = df_result["tp"].sum()
    total_fp = df_result["fp"].sum()
    total_fn = df_result["fn"].sum()
    micro_p  = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = round(2 * micro_p * micro_r / (micro_p + micro_r), 4) if (micro_p + micro_r) > 0 else 0.0

    titulo = f"Graph comparison: {dataset_name}" if dataset_name else "Graph comparison"
    print(f"\n── {titulo} {'─' * max(0, 55 - len(titulo))}")

    df_diff = df_result[(df_result["fp"] > 0) | (df_result["fn"] > 0)]
    n_perfect = len(df_result) - len(df_diff)
    print(f"  Graphs with differences: {len(df_diff)}/{len(df_result)}  ({n_perfect} perfect)\n")

    # Columnas métricas (sin las aristas) para el tabulate
    cols_tabla = ["text_id", "tp", "fp", "fn", "precision", "recall", "f1"]
    print(tabulate(df_diff[cols_tabla], headers="keys", tablefmt="rounded_outline", showindex=False))

    # Detalle de aristas erróneas por texto
    for _, row in df_diff.iterrows():
        print(f"\n  {row['text_id']}")
        for src, tgt, lbl in sorted(row["fp_edges"]):
            print(f"    FP  {src} ──[{lbl}]──> {tgt}")
        for src, tgt, lbl in sorted(row["fn_edges"]):
            print(f"    FN  {src} ──[{lbl}]──> {tgt}")

    print(f"\n  Micro-avg  →  Precision: {micro_p}  Recall: {micro_r}  F1: {micro_f1}")
    print(f"  (TP={total_tp}, FP={total_fp}, FN={total_fn}  over {len(text_ids)} texts)\n")

    return df_result


# ── Usage ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TRUE_GRAPHS_PATH    = "./results/minimal_v5/"
    PRED_GRAPHS_PATH    = "./results/predictions_v5/"        

    # datasets = ["AAEC", "CDCP", "ArGPT", "MTC", "AASD"]
    datasets = ["AAEC"]

    metrics_report(graphs_by_dataset)

    for name in datasets:
        df_true = pd.read_pickle(TRUE_GRAPHS_PATH + f"{name}_Minimal_Grafo.pkl")
        df_pred = pd.read_pickle(PRED_GRAPHS_PATH   + f"{name}_Predicted_Grafo.pkl")
        report_comparison(df_true, df_pred, dataset_name=name)




"""
ground-truth analysis:
╭──────────┬──────────────────┬────────────┬─────────────┬──────────┬────────────┬────────────────────────┬──────────╮
│          │   total_text_arg │   isolados │ subgrafos   │   conexo │   aciclico │ nos_raiz               │   arvore │
├──────────┼──────────────────┼────────────┼─────────────┼──────────┼────────────┼────────────────────────┼──────────┤
│ AAEC     │              402 │          2 │             │      400 │        400 │ 1: 71, 2: 311, 3: 18   │       71 │

predicted analysis:
╭──────┬─────────┬────────────┬─────────────┬───────────────┬───────┬────────┬────────────────┬──────────────────────╮
│      │   total │   isolates │   subgraphs │   weakly_conn │   dag │   tree │   arborescence │ roots                │
├──────┼─────────┼────────────┼─────────────┼───────────────┼───────┼────────┼────────────────┼──────────────────────┤
│ AAEC │     402 │          2 │           0 │           400 │   402 │     71 │             71 │ 1: 71, 2: 312, 3: 19 │
╰──────┴─────────┴────────────┴─────────────┴───────────────┴───────┴────────┴────────────────┴──────────────────────╯

The difference between acyclic graphs (400) and DAGs (402) arises because the file `analisis_dataset_minimal_v5.py` 
applies an initial filter for connected graphs, excluding 2 graphs from subsequent analyses.


 Graphs with differences: 58/402  (344 perfect)

╭───────────┬──────┬──────┬──────┬─────────────┬──────────┬────────╮
│ text_id   │   tp │   fp │   fn │   precision │   recall │     f1 │
├───────────┼──────┼──────┼──────┼─────────────┼──────────┼────────┤
│ AAEC_016  │   13 │    1 │    1 │      0.9286 │   0.9286 │ 0.9286 │
│ AAEC_021  │   18 │    2 │    2 │      0.9    │   0.9    │ 0.9    │
│ AAEC_028  │   16 │    1 │    1 │      0.9412 │   0.9412 │ 0.9412 │
│ AAEC_036  │   14 │    1 │    1 │      0.9333 │   0.9333 │ 0.9333 │
│ AAEC_042  │   11 │    1 │    1 │      0.9167 │   0.9167 │ 0.9167 │
│ AAEC_052  │   11 │    2 │    2 │      0.8462 │   0.8462 │ 0.8462 │
│ AAEC_058  │   19 │    1 │    1 │      0.95   │   0.95   │ 0.95   │
│ AAEC_061  │   22 │    1 │    1 │      0.9565 │   0.9565 │ 0.9565 │
│ AAEC_062  │   19 │    1 │    1 │      0.95   │   0.95   │ 0.95   │
│ AAEC_067  │   12 │    1 │    1 │      0.9231 │   0.9231 │ 0.9231 │
│ AAEC_071  │    8 │    2 │    2 │      0.8    │   0.8    │ 0.8    │
│ AAEC_072  │   14 │    1 │    1 │      0.9333 │   0.9333 │ 0.9333 │
│ AAEC_073  │   13 │    1 │    1 │      0.9286 │   0.9286 │ 0.9286 │
│ AAEC_086  │   17 │    1 │    1 │      0.9444 │   0.9444 │ 0.9444 │
│ AAEC_087  │   15 │    1 │    1 │      0.9375 │   0.9375 │ 0.9375 │
│ AAEC_089  │   19 │    1 │    1 │      0.95   │   0.95   │ 0.95   │
│ AAEC_097  │   18 │    1 │    1 │      0.9474 │   0.9474 │ 0.9474 │
│ AAEC_103  │    6 │    1 │    1 │      0.8571 │   0.8571 │ 0.8571 │
│ AAEC_112  │   11 │    1 │    1 │      0.9167 │   0.9167 │ 0.9167 │
│ AAEC_117  │   13 │    1 │    1 │      0.9286 │   0.9286 │ 0.9286 │
│ AAEC_118  │   13 │    1 │    1 │      0.9286 │   0.9286 │ 0.9286 │
│ AAEC_129  │   18 │    3 │    3 │      0.8571 │   0.8571 │ 0.8571 │
│ AAEC_139  │   12 │    2 │    2 │      0.8571 │   0.8571 │ 0.8571 │
│ AAEC_152  │   12 │    1 │    1 │      0.9231 │   0.9231 │ 0.9231 │
│ AAEC_160  │   10 │    4 │    4 │      0.7143 │   0.7143 │ 0.7143 │
│ AAEC_162  │    9 │    2 │    2 │      0.8182 │   0.8182 │ 0.8182 │
│ AAEC_163  │   17 │    1 │    1 │      0.9444 │   0.9444 │ 0.9444 │
│ AAEC_169  │   31 │    1 │    1 │      0.9688 │   0.9688 │ 0.9688 │
│ AAEC_172  │   21 │    2 │    2 │      0.913  │   0.913  │ 0.913  │
│ AAEC_187  │   14 │    2 │    2 │      0.875  │   0.875  │ 0.875  │
│ AAEC_192  │   12 │    1 │    1 │      0.9231 │   0.9231 │ 0.9231 │
│ AAEC_197  │   12 │    2 │    2 │      0.8571 │   0.8571 │ 0.8571 │
│ AAEC_211  │   17 │    1 │    1 │      0.9444 │   0.9444 │ 0.9444 │
│ AAEC_212  │   15 │    2 │    2 │      0.8824 │   0.8824 │ 0.8824 │
│ AAEC_225  │   17 │    3 │    3 │      0.85   │   0.85   │ 0.85   │
│ AAEC_227  │   12 │    2 │    2 │      0.8571 │   0.8571 │ 0.8571 │
│ AAEC_229  │   13 │    6 │    6 │      0.6842 │   0.6842 │ 0.6842 │
│ AAEC_232  │   12 │    2 │    2 │      0.8571 │   0.8571 │ 0.8571 │
│ AAEC_234  │   20 │    1 │    1 │      0.9524 │   0.9524 │ 0.9524 │
│ AAEC_255  │   14 │    2 │    2 │      0.875  │   0.875  │ 0.875  │
│ AAEC_266  │   16 │    2 │    2 │      0.8889 │   0.8889 │ 0.8889 │
│ AAEC_293  │   24 │    1 │    1 │      0.96   │   0.96   │ 0.96   │
│ AAEC_295  │   14 │    2 │    2 │      0.875  │   0.875  │ 0.875  │
│ AAEC_323  │   14 │    1 │    1 │      0.9333 │   0.9333 │ 0.9333 │
│ AAEC_330  │   13 │    1 │    1 │      0.9286 │   0.9286 │ 0.9286 │
│ AAEC_331  │   20 │    2 │    2 │      0.9091 │   0.9091 │ 0.9091 │
│ AAEC_335  │    4 │    3 │    3 │      0.5714 │   0.5714 │ 0.5714 │
│ AAEC_338  │   12 │    1 │    1 │      0.9231 │   0.9231 │ 0.9231 │
│ AAEC_339  │    7 │    2 │    2 │      0.7778 │   0.7778 │ 0.7778 │
│ AAEC_341  │   14 │    2 │    2 │      0.875  │   0.875  │ 0.875  │
│ AAEC_343  │    7 │    2 │    2 │      0.7778 │   0.7778 │ 0.7778 │
│ AAEC_347  │   13 │    1 │    1 │      0.9286 │   0.9286 │ 0.9286 │
│ AAEC_355  │   11 │    2 │    2 │      0.8462 │   0.8462 │ 0.8462 │
│ AAEC_359  │   16 │    2 │    2 │      0.8889 │   0.8889 │ 0.8889 │
│ AAEC_373  │   23 │    1 │    1 │      0.9583 │   0.9583 │ 0.9583 │
│ AAEC_379  │   10 │    2 │    2 │      0.8333 │   0.8333 │ 0.8333 │
│ AAEC_395  │   13 │    1 │    1 │      0.9286 │   0.9286 │ 0.9286 │
│ AAEC_399  │   10 │    1 │    1 │      0.9091 │   0.9091 │ 0.9091 │
╰───────────┴──────┴──────┴──────┴─────────────┴──────────┴────────╯

"""

