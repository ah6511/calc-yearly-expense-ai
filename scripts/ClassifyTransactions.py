"""
Cluster bank transactions by description using embeddings + KMeans,
with automatic best-k selection and optional 2D visualization.

Features:
  - Load CSV -> DataFrame
  - Embeddings: OpenAI (API) or local (sentence-transformers)
  - KMeans clustering with silhouette-based best-k search
  - Optional 2D plot (PCA by default; UMAP if available)
  - Writes labeled CSV; optional simple cluster report

Example (OpenAI):
  export OPENAI_API_KEY="your_api_key"
  python cluster_transactions_adv.py \
    --csv_path ./transactions.csv \
    --description_col description \
    --embedding_provider openai \
    --openai_model text-embedding-3-small \
    --k_min 8 --k_max 20 \
    --out_path ./transactions_labeled.csv \
    --plot_path ./clusters_pca.png

Example (Local, offline):
  python cluster_transactions_adv.py \
    --csv_path ./transactions.csv \
    --description_col description \
    --embedding_provider local \
    --local_model all-MiniLM-L6-v2 \
    --k_min 8 --k_max 20 \
    --out_path ./transactions_labeled.csv \
    --plot_path ./clusters_umap.png --use_umap
"""

import os
import sys
import time
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")  # headless environments
import matplotlib.pyplot as plt

# Optional imports loaded lazily
_HAVE_OPENAI = False
_HAVE_SENTENCE_TRANSFORMERS = False
_HAVE_UMAP = False

try:
    from openai import OpenAI  # pip install openai>=1.0.0
    _HAVE_OPENAI = True
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
    _HAVE_SENTENCE_TRANSFORMERS = True
except Exception:
    pass

try:
    import umap  # pip install umap-learn
    _HAVE_UMAP = True
except Exception:
    _HAVE_UMAP = False


def load_dataframe(csv_path: str,
                   description_col: str = "description") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if description_col not in df.columns:
        raise ValueError(f"Missing required column '{description_col}' in CSV.")
    # Create a cleaned description column used for embedding
    df["_desc_clean"] = (
        df[description_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    # Drop empty descriptions
    df = df[df["_desc_clean"].str.len() > 0].reset_index(drop=True)
    return df


def embed_texts_openai(texts: List[str],
                       model: str = "text-embedding-3-small",
                       batch_size: int = 100,
                       max_retries: int = 5,
                       retry_backoff_sec: float = 2.0) -> List[List[float]]:
    if not _HAVE_OPENAI:
        raise RuntimeError("OpenAI SDK not installed. pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                embeddings.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                sleep_time = retry_backoff_sec * (2 ** attempt)
                print(f"[OpenAI] Retry {attempt+1}/{max_retries} in {sleep_time:.1f}s: {e}")
                time.sleep(sleep_time)
    return embeddings


def embed_texts_local(texts: List[str],
                      local_model: str = "all-MiniLM-L6-v2",
                      device: str = "cpu",
                      batch_size: int = 64) -> List[List[float]]:
    if not _HAVE_SENTENCE_TRANSFORMERS:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
    model = SentenceTransformer(local_model, device=device)
    # normalize_embeddings=False here; we do normalization before KMeans
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False)
    return vecs.tolist()


def choose_best_k(X: np.ndarray,
                  k_min: int,
                  k_max: int,
                  random_state: int = 42,
                  n_init: int = 10) -> Tuple[int, float]:
    """
    Returns (best_k, best_silhouette). Uses euclidean on L2-normalized embeddings (cosine-like).
    """
    best_k, best_score = None, -1.0
    for k in range(max(2, k_min), max(3, k_max + 1)):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, metric="euclidean")
        print(f"  k={k:>3} silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    if best_k is None:
        raise RuntimeError("Failed to select best k.")
    print(f"Selected k={best_k} (silhouette={best_score:.4f})")
    return best_k, best_score


def cluster_embeddings(embeddings: np.ndarray,
                       n_clusters: int,
                       random_state: int = 42,
                       n_init: int = 10) -> Tuple[np.ndarray, KMeans, np.ndarray]:
    """
    L2-normalize embeddings, run KMeans, return (labels, model, normalized_X).
    """
    X = normalize(embeddings, norm="l2")
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km, X


def plot_2d(X: np.ndarray,
            labels: np.ndarray,
            path: str,
            title: str = "Transaction clusters (2D)",
            use_umap: bool = False,
            random_state: int = 42) -> None:
    if use_umap and _HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        Z = reducer.fit_transform(X)
        subtitle = "UMAP"
    else:
        pca = PCA(n_components=2, random_state=random_state)
        Z = pca.fit_transform(X)
        subtitle = "PCA"

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=12, cmap="tab20", alpha=0.75, edgecolors="none")
    plt.title(f"{title} ({subtitle})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Wrote plot to: {path}")


def write_cluster_report(df: pd.DataFrame,
                         labels: np.ndarray,
                         X: np.ndarray,
                         description_col: str,
                         report_path: str,
                         samples_per_cluster: int = 5) -> None:
    """
    Writes a simple TSV report with cluster sizes and representative samples (nearest to centroid).
    """
    df = df.copy()
    df["cluster_label"] = labels
    # Find simple representatives by nearest to cluster mean in 2D (PCA) to keep it light
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    df["_z0"], df["_z1"] = Z[:, 0], Z[:, 1]

    lines = []
    for cid in sorted(df["cluster_label"].unique()):
        sub = df[df["cluster_label"] == cid].copy()
        # centroid in 2D
        c0, c1 = sub["_z0"].mean(), sub["_z1"].mean()
        sub["_dist"] = np.sqrt((sub["_z0"] - c0) ** 2 + (sub["_z1"] - c1) ** 2)
        reps = sub.nsmallest(samples_per_cluster, "_dist")
        lines.append(f"# Cluster {cid} (size={len(sub)})")
        for _, r in reps.iterrows():
            desc = str(r[description_col])[:200].replace("\n", " ")
            lines.append(f"- {desc}")
        lines.append("")  # blank line

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote cluster report to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster bank transactions by description using embeddings + KMeans.")
    parser.add_argument("--csv_path", required=True, help="Path to input CSV with transactions.")
    parser.add_argument("--description_col", default="description", help="Column containing transaction descriptions.")
    parser.add_argument("--embedding_provider", choices=["openai", "local"], default="openai", help="Embedding backend.")
    parser.add_argument("--openai_model", default="text-embedding-3-small", help="OpenAI embedding model.")
    parser.add_argument("--local_model", default="all-MiniLM-L6-v2", help="sentence-transformers model or local path.")
    parser.add_argument("--device", default="cpu", help="Device for local model: cpu or cuda.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for embedding calls.")
    parser.add_argument("--n_clusters", type=int, default=10, help="If set with --auto_k off, use this k.")
    parser.add_argument("--auto_k", action="store_true", help="If set, search k in [k_min, k_max] via silhouette.")
    parser.add_argument("--k_min", type=int, default=8, help="Min k for auto search (>=2).")
    parser.add_argument("--k_max", type=int, default=20, help="Max k for auto search.")
    parser.add_argument("--plot_path", default=None, help="If set, write a 2D plot (PCA or UMAP).")
    parser.add_argument("--use_umap", action="store_true", help="Use UMAP for 2D projection if available.")
    parser.add_argument("--report_path", default=None, help="If set, write a simple cluster report TSV.")
    parser.add_argument("--samples_per_cluster", type=int, default=5, help="Samples per cluster in report.")
    parser.add_argument("--out_path", default="./transactions_labeled.csv", help="Output CSV with cluster labels.")
    args = parser.parse_args()

    print("Loading data...")
    df = load_dataframe(csv_path=args.csv_path, description_col=args.description_col)
    texts = df["_desc_clean"].tolist()

    print(f"Embedding {len(df)} descriptions using provider '{args.embedding_provider}'...")
    if args.embedding_provider == "openai":
        vectors = embed_texts_openai(texts, model=args.openai_model, batch_size=args.batch_size)
    else:
        vectors = embed_texts_local(texts, local_model=args.local_model, device=args.device, batch_size=max(8, args.batch_size // 2))

    emb = np.array(vectors, dtype=np.float32)
    X = normalize(emb, norm="l2")  # cosine-like geometry

    if args.auto_k:
        print(f"Selecting best k in [{args.k_min}, {args.k_max}] via silhouette score...")
        best_k, _ = choose_best_k(X, k_min=args.k_min, k_max=args.k_max, random_state=42)
        k = best_k
    else:
        k = args.n_clusters
        print(f"Using specified k={k}")

    print(f"Clustering into {k} clusters...")
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    df["cluster_label"] = labels

    # Quick counts per cluster
    counts = df["cluster_label"].value_counts().sort_index()
    print("Cluster sizes:")
    for cid, cnt in counts.items():
        print(f"  Cluster {cid}: {cnt}")

    if args.plot_path:
        try:
            plot_2d(X, labels, path=args.plot_path, use_umap=args.use_umap)
        except Exception as e:
            print(f"Plotting failed: {e}", file=sys.stderr)

    if args.report_path:
        try:
            write_cluster_report(df, labels, X, args.description_col, args.report_path, args.samples_per_cluster)
        except Exception as e:
            print(f"Report generation failed: {e}", file=sys.stderr)

    print(f"Writing labeled data to: {args.out_path}")
    # Keep original columns
    out_df = df.drop(columns=["_desc_clean"], errors="ignore")
    out_df.to_csv(args.out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
