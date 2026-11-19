# =====================================================================
#  Title:           Analize of dataframe .pkl created to each dataset
#  Description:     Is a secod step, it create a summary dataframe
#  Version:         V6
#  Author:          cbarreto
# =====================================================================

import pandas as pd
from tqdm import tqdm

result_path = "./../results/minimal_v6/"
name_dataframe_summary = "dataframe_summary"

# Cargar el DataFrame
datasets = {
    "CDCP":     "CDCP_Minimal_Grafo.pkl",
    "AAEC":     "AAEC_Minimal_Grafo.pkl",
    "ArGPT":    "ArGPT_Minimal_Grafo.pkl",
    "MTC":      "MTC_Minimal_Grafo.pkl",
    "AASD":     "AASD_Minimal_Grafo.pkl",
    "Glaucoma": "AbstRCT_glaucoma_Minimal_Grafo.pkl",
    "Mixed":    "AbstRCT_mixed_Minimal_Grafo.pkl",
    "Neoplasm": "AbstRCT_neoplasm_Minimal_Grafo.pkl",
    "ACSP":     "ACSP_Minimal_Grafo.pkl",
    "AMP":      "AMP_Minimal_Grafo.pkl",
    "USD":      "USD_Minimal_Grafo.pkl",
}

# Lista de Dataframes
dfs = {}
for name, file in datasets.items():
    df = pd.read_pickle(result_path + file)
    dfs[name] = df

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

# Caracteristicas do grafo
comp = "n_comp"
nodes_roots = "n_root"
acyclic = "is_acyclic"
connected = "is_connected"
is_tree = "is_tree"
is_arborescence = "is_arborescence"
is_floresta = "is_forest"


for index, (key, df) in tqdm(enumerate(dfs.items()), total=len(dfs)):
    dfs_indices[index] = key
    total_text_arg  = len(df)
    isolados = ((df["n_isolates"] != 0)).sum()
    subgrafos = df[comp].value_counts().sort_index().to_dict()
    nos_raiz = df[nodes_roots].value_counts().sort_index().to_dict()
    aciclico = df[acyclic].value_counts().sort_index().to_dict()
    conexo = df[connected].value_counts().sort_index().to_dict()
    arvore = df[is_tree].value_counts().sort_index().to_dict()
    arborecente = df[is_arborescence].value_counts().sort_index().to_dict()
    floresta = df[is_floresta].value_counts().sort_index().to_dict()
    
    dfs_sumary.append({
        "total_text_arg":total_text_arg,
        "no_isolados":isolados,        
        "no_componet": ", ".join(f"{v}: {k}" for k, v in subgrafos.items()),
        "nos_raiz": ", ".join(f"{v}: {k}" for k, v in nos_raiz.items()),
        "aciclico": ", ".join(f"{v}: {k}" for k, v in aciclico.items()),
        "conexo": ", ".join(f"{v}: {k}" for k, v in conexo.items()),
        "arvore": ", ".join(f"{v}: {k}" for k, v in arvore.items()),
        "arborecente": ", ".join(f"{v}: {k}" for k, v in arborecente.items()),
        "floresta": ", ".join(f"{v}: {k}" for k, v in floresta.items()),
        }
    )


df_resultados = pd.json_normalize(dfs_sumary)
df_resultados = df_resultados.rename(index=dfs_indices)
df_resultados.to_pickle(result_path + name_dataframe_summary + ".pkl")
df_resultados.to_csv(result_path + name_dataframe_summary + ".csv", index=True)