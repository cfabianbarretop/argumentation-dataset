# -*- coding: utf-8 -*-
"""
build_3relation_dataset.py
──────────────────────────
Matches link_<split>.csv against relation_<split>.csv to:

    1. Report conflicts:
        1.1 Verify whether the number of pairs in `link_<split>` and `relation_<split>` matches. 
            If not, report how many and which pairs differ. 
        1.2 Verify whether there are any pairs that appear as `None` in `link_<split>` but are labeled as `Support` or `Attack` in `relation_<split>`. 
            Report how many and which pairs these are. 
        1.3 Verify whether there are any pairs that appear as `Link` in `link_<split>` but are labeled as `None` in `relation_<split>`. 
            Report how many and which pairs these are. 

    2. Create a `relation3_<split>.csv` file by augmenting `relation_<split>` with the `None` pairs from `link_<split>`. 
        This results in a file for ternary classification (support | attack | none).

Two input CSV files are expected for each split:
    - link_<split>.csv
        text_id: str
        source_id: str
        source_text: str
        target_id: str
        target_text: str
        label: str
        minimalist_labels: str, values: [Link, None]

    - relation_<split>.csv
        text_id: str
        source_id: str
        source_text: str
        target_id: str
        target_text: str
        label: str
        minimalist_labels: str, values: [support, attack]
"""

import argparse
import os
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "../dataset_minimal/"

# Key columns that uniquely identify a directed pair
PAIR_KEYS = ["text_id", "source_id", "target_id"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_relation(split: str, file_path: str) -> pd.DataFrame:
    path = os.path.join(file_path, f"relation_{split}.csv")
    df = pd.read_csv(path, header=0, keep_default_na=False)
    print(f"[{split}] relation  → {df.shape[0]:>6} rows | loaded from: {path}")
    return df


def load_link(split: str, file_path: str) -> pd.DataFrame:
    path = os.path.join(file_path, f"link_{split}.csv")
    df = pd.read_csv(path, header=0, keep_default_na=False)
    print(f"[{split}] link      → {df.shape[0]:>6} rows | loaded from: {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def report_conflicts(link_df: pd.DataFrame, relation_df: pd.DataFrame, split: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  CONFLICT REPORT — {split}")
    print(f"{'─'*60}")

    link_df["minimalist_labels"] = link_df["minimalist_labels"].str.strip()
    relation_df["minimalist_labels"] = relation_df["minimalist_labels"].str.strip()

    link_linked = link_df[link_df["minimalist_labels"].str.lower() == "link"]
    link_none   = link_df[link_df["minimalist_labels"].str.lower() == "none"]

    rel_support    = relation_df[relation_df["minimalist_labels"].str.lower() == "support"]
    rel_attack    = relation_df[relation_df["minimalist_labels"].str.lower() == "attack"]

    link_linked_pairs = set(map(tuple, link_linked[PAIR_KEYS].values))
    link_none_pairs   = set(map(tuple, link_none[PAIR_KEYS].values))
    rel_pairs = set(map(tuple, relation_df[PAIR_KEYS].values))
    rel_support_pairs = set(map(tuple, rel_support[PAIR_KEYS].values))
    rel_attack_pairs = set(map(tuple, rel_attack[PAIR_KEYS].values))

    # ── 1.1 ───────────────────────────────────────────────────────────────
    print(f"\n[1.1] Pair counts")
    print(f"      link file  → total: {len(link_df):>5}  "
          f"(Link: {len(link_linked_pairs)}, None: {len(link_none_pairs)})")
    print(f"      relation file → total: {len(relation_df):>5}  "
          f"(Support: {len(rel_support_pairs)}, Attack: {len(rel_attack_pairs)})")

    only_in_link     = link_linked_pairs - rel_pairs   # Link en link pero ausente en relation
    only_in_relation = rel_pairs - link_linked_pairs   # en relation pero ausente como Link en link

    if not only_in_link and not only_in_relation:
        print("      ✓ All 'Link' pairs in link are present in relation and vice versa.")
    else:
        if only_in_link:
            print(f"\n      ✗ {len(only_in_link)} pair(s) labeled 'Link' in link but MISSING in relation:")
            # for p in sorted(only_in_link):
            #     print(f"        {p}")
        if only_in_relation:
            print(f"\n      ✗ {len(only_in_relation)} pair(s) in relation but NOT labeled 'Link' in link:")
            for p in sorted(only_in_relation):
                print(f"        {p}")

    # ── Merge para 1.2 y 1.3 ───────────────────────────────
    merged = link_df[PAIR_KEYS + ["minimalist_labels"]].merge(
        relation_df[PAIR_KEYS + ["minimalist_labels"]],
        on=PAIR_KEYS,
        how="inner",
        suffixes=("_link", "_relation"),
    )

    # ── 1.2 None en link csv pero Support/Attack en relation csv ──────────────────
    conflict_12 = merged[
        merged["minimalist_labels_link"].str.lower() == "none"
    ]
    print(f"\n[1.2] Pairs labeled 'None' in link but present in relation "
          f"(as support/attack): {len(conflict_12)}")
    if not conflict_12.empty:
        print("      ✗ Conflicting pairs:")
        for _, row in conflict_12.iterrows():
            print(f"        {tuple(row[k] for k in PAIR_KEYS)}  "
                  f"link={row['minimalist_labels_link']!r}  "
                  f"relation={row['minimalist_labels_relation']!r}")
    else:
        print("      ✓ No conflicts found.")

    # ── 1.3 Link en link pero None en relation ────────────────────────────
    conflict_13 = merged[
        (merged["minimalist_labels_link"].str.lower() == "link") &
        (merged["minimalist_labels_relation"].str.lower() == "none")
    ]
    print(f"\n[1.3] Pairs labeled 'Link' in link but 'None' in relation: "
          f"{len(conflict_13)}")
    if not conflict_13.empty:
        print("      ✗ Conflicting pairs:")
        for _, row in conflict_13.iterrows():
            print(f"        {tuple(row[k] for k in PAIR_KEYS)}  "
                  f"link={row['minimalist_labels_link']!r}  "
                  f"relation={row['minimalist_labels_relation']!r}")
    else:
        print("      ✓ No conflicts found.")

    print(f"\n{'─'*60}\n")

def build_relation3(link_df: pd.DataFrame, relation_df: pd.DataFrame, split: str, file_path: str) -> None:
    """
    2. Augment relation_<split> with the None pairs from link_<split>
       to create relation3_<split>.csv for ternary classification (support | attack | none).

    Conflict resolution:
        - 1.1 inconsistencies (in relation but absent in link) → removed from relation
        - 1.2 inconsistencies (None in link but present in relation) → kept with link info (label=none)
    """

    link_linked_pairs = set(map(tuple,
        link_df[link_df["minimalist_labels"].str.lower() == "link"][PAIR_KEYS].values
    ))
    rel_pairs = set(map(tuple, relation_df[PAIR_KEYS].values))

    # ── 1.1: pares en relation sin respaldo en link → eliminar ────────────
    inconsistent_11 = rel_pairs - link_linked_pairs
    relation_clean = relation_df[
        ~relation_df[PAIR_KEYS].apply(tuple, axis=1).isin(inconsistent_11)
    ].copy()

    # ── 1.2: pares None en link que están en relation → conservar como none
    #    Se excluyen de relation_clean y se dejan caer en none_pairs (ya tienen label=none)
    none_df = link_df[link_df["minimalist_labels"].str.lower() == "none"].copy()
    none_pairs_keys = set(map(tuple, none_df[PAIR_KEYS].values))

    inconsistent_12 = rel_pairs & none_pairs_keys  # intersección: None en link Y en relation
    relation_clean = relation_clean[
        ~relation_clean[PAIR_KEYS].apply(tuple, axis=1).isin(inconsistent_12)
    ].copy()

    # ── Preparar none_pairs con etiqueta normalizada ───────────────────────
    none_pairs = none_df.copy()
    none_pairs["label"] = "none"
    none_pairs["minimalist_labels"] = "none"

    # ── Concatenar ─────────────────────────────────────────────────────────
    relation3_df = pd.concat([relation_clean, none_pairs], ignore_index=True)

    # ── Guardar ────────────────────────────────────────────────────────────
    output_path = os.path.join(file_path, f"relation3_{split}.csv")
    relation3_df.to_csv(output_path, index=False)

    # ── Reporte ────────────────────────────────────────────────────────────
    print(f"\n[2] relation3_{split}.csv created")
    print(f"    relation rows (original)       : {len(relation_df):>6}")
    if inconsistent_11:
        print(f"    removed (1.1 inconsistencies)  : {len(inconsistent_11):>6}")
        for p in sorted(inconsistent_11):
            print(f"      - {p}")
    if inconsistent_12:
        print(f"    replaced by link info (1.2)    : {len(inconsistent_12):>6}")
        for p in sorted(inconsistent_12):
            print(f"      - {p}")
    print(f"    relation rows (clean)          : {len(relation_clean):>6}  (support/attack)")
    print(f"    none rows added                : {len(none_pairs):>6}  (from link)")
    print(f"    total rows                     : {len(relation3_df):>6}")
    print(f"    saved to                       : {output_path}")
    print(f"    label distribution:\n{relation3_df['minimalist_labels'].value_counts().to_string()}")

def build_relation3_old(link_df: pd.DataFrame, relation_df: pd.DataFrame, split: str, file_path: str) -> None:
    """
    2. Augment relation_<split> with the None pairs from link_<split>
       to create relation3_<split>.csv for ternary classification (support | attack | none).
       Pairs flagged as inconsistent in 1.1 are excluded.
    """

    link_linked_pairs = set(map(tuple, 
        link_df[link_df["minimalist_labels"].str.lower() == "link"][PAIR_KEYS].values
    ))
    rel_pairs = set(map(tuple, relation_df[PAIR_KEYS].values))

    # Pares inconsistentes: en relation pero no etiquetados 'Link' en link
    inconsistent_pairs = rel_pairs - link_linked_pairs

    # Filtrar relation eliminando inconsistentes
    relation_clean = relation_df[
        ~relation_df[PAIR_KEYS].apply(tuple, axis=1).isin(inconsistent_pairs)
    ].copy()

    # Extraer pares None de link
    none_pairs = link_df[link_df["minimalist_labels"].str.lower() == "none"].copy()
    none_pairs["minimalist_labels"] = "none"

    # Concatenar
    relation3_df = pd.concat([relation_clean, none_pairs], ignore_index=True)

    # Guardar
    output_path = os.path.join(file_path, f"relation3_{split}.csv")
    relation3_df.to_csv(output_path, index=False)

    print(f"\n[2] relation3_{split}.csv created")
    print(f"    relation rows (original) : {len(relation_df):>6}")
    if inconsistent_pairs:
        print(f"    inconsistent rows removed: {len(inconsistent_pairs):>6}")
        for p in sorted(inconsistent_pairs):
            print(f"      removed: {p}")
    print(f"    relation rows (clean)    : {len(relation_clean):>6}  (support/attack)")
    print(f"    none rows added          : {len(none_pairs):>6}  (from link)")
    print(f"    total rows               : {len(relation3_df):>6}")
    print(f"    saved to                 : {output_path}")
    print(f"    label distribution:\n{relation3_df['minimalist_labels'].value_counts().to_string()}")
    

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val"]) # ["val", "train", "test"]
    parser.add_argument("--file_path", default=DATA_DIR)
    args = parser.parse_args()

    for split in args.splits:
        print(f"\nProcessing split: {split}")
        relation_df = load_relation(split, args.file_path)
        link_df     = load_link(split, args.file_path)

        report_conflicts(link_df, relation_df, split)
        build_relation3(link_df, relation_df, split, args.file_path)

    print("\nDone.")


if __name__ == "__main__":
    """
    Usage examples:
        python build_3relation_dataset.py --splits val train
        python build_3relation_dataset.py --splits val --file_path ./dataset/
    """
    main()