"""Cluster bank transactions by description using embeddings + KMeans.

Features:
  - Load CSV -> DataFrame
  - Embeddings: OpenAI (API) or local (sentence-transformers)
  - KMeans clustering with silhouette-based best-k search
  - Optional 2D plot (PCA by default; UMAP if available)
  - Writes labeled CSV; optional simple cluster report

Requires the `analysis` extra (`uv sync --extra analysis`). `openai` is installed
by default; `sentence-transformers` and `umap-learn` are optional.

Example (OpenAI):
  uv run python scripts/ClassifyTransactions.py \
    --csv_path transactions/total_withdrawals.csv \
    --embedding_provider openai \
    --auto_k --k_min 8 --k_max 20 \
    --out_path transactions/transactions_labeled.csv \
    --plot_path clusters_pca.png

Example (local, offline):
  uv run python scripts/ClassifyTransactions.py \
    --csv_path transactions/total_withdrawals.csv \
    --embedding_provider local \
    --auto_k --k_min 8 --k_max 20 \
    --out_path transactions/transactions_labeled.csv \
    --plot_path clusters_umap.png --use_umap
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # headless environments

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

logger = logging.getLogger("classify_transactions")

RANDOM_STATE = 42
KMEANS_N_INIT = 10
CLEAN_COL = "_desc_clean"
LABEL_COL = "cluster_label"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dataframe(csv_path: str, description_col: str = "description") -> pd.DataFrame:
    """Load transactions and add a cleaned description column used for embedding."""
    df = pd.read_csv(csv_path)
    if description_col not in df.columns:
        raise ValueError(f"Missing required column '{description_col}' in CSV.")
    df[CLEAN_COL] = df[description_col].astype(str).str.strip().str.lower()
    return df[df[CLEAN_COL].str.len() > 0].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def embed_texts_openai(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_retries: int = 5,
    retry_backoff_sec: float = 2.0,
) -> list[list[float]]:
    """Embed texts with the OpenAI embeddings API, retrying with exponential backoff."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK not installed. Run: uv add openai") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OSError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=model, input=batch)
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                sleep_time = retry_backoff_sec * (2**attempt)
                logger.warning(
                    "[OpenAI] Retry %d/%d in %.1fs: %s", attempt + 1, max_retries, sleep_time, exc
                )
                time.sleep(sleep_time)
            else:
                embeddings.extend(d.embedding for d in resp.data)
                break
    return embeddings


def embed_texts_local(
    texts: list[str],
    local_model: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed texts locally with sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers not installed. Run: uv add sentence-transformers"
        ) from exc

    model = SentenceTransformer(local_model, device=device)
    # Normalization happens once before clustering, not here.
    vecs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False
    )
    return vecs.tolist()


def embed_texts(texts: list[str], args: argparse.Namespace) -> np.ndarray:
    """Embed texts with the selected provider and return L2-normalized vectors."""
    if args.embedding_provider == "openai":
        vectors = embed_texts_openai(texts, model=args.openai_model, batch_size=args.batch_size)
    else:
        vectors = embed_texts_local(
            texts,
            local_model=args.local_model,
            device=args.device,
            batch_size=max(8, args.batch_size // 2),
        )
    # L2-normalize so euclidean KMeans behaves like cosine similarity.
    return normalize(np.array(vectors, dtype=np.float32), norm="l2")


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def fit_kmeans(X: np.ndarray, n_clusters: int) -> tuple[np.ndarray, KMeans]:
    """Fit KMeans on already-normalized embeddings and return (labels, model)."""
    km = KMeans(n_clusters=n_clusters, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
    return km.fit_predict(X), km


def choose_best_k(X: np.ndarray, k_min: int, k_max: int) -> tuple[int, float]:
    """Return (best_k, best_silhouette) searched over [k_min, k_max]."""
    best_k, best_score = None, -1.0
    for k in range(max(2, k_min), max(3, k_max + 1)):
        labels, _ = fit_kmeans(X, k)
        score = float(silhouette_score(X, labels, metric="euclidean"))
        logger.info("  k=%3d silhouette=%.4f", k, score)
        if score > best_score:
            best_k, best_score = k, score
    if best_k is None:
        raise RuntimeError("Failed to select best k.")
    logger.info("Selected k=%d (silhouette=%.4f)", best_k, best_score)
    return best_k, best_score


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def project_2d(X: np.ndarray, use_umap: bool = False) -> tuple[np.ndarray, str]:
    """Project embeddings to 2D with UMAP (if requested and installed) or PCA."""
    if use_umap:
        try:
            import umap
        except ImportError:
            logger.warning("umap-learn not installed; falling back to PCA.")
        else:
            reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
            return reducer.fit_transform(X), "UMAP"
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X), "PCA"


def plot_2d(
    X: np.ndarray,
    labels: np.ndarray,
    path: str,
    title: str = "Transaction clusters (2D)",
    use_umap: bool = False,
) -> None:
    Z, method = project_2d(X, use_umap=use_umap)
    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=12, cmap="tab20", alpha=0.75, edgecolors="none")
    plt.title(f"{title} ({method})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    logger.info("Wrote plot to: %s", path)


def write_cluster_report(
    df: pd.DataFrame,
    labels: np.ndarray,
    X: np.ndarray,
    description_col: str,
    report_path: str,
    samples_per_cluster: int = 5,
) -> None:
    """Write a text report with cluster sizes and samples nearest each cluster's 2D centroid."""
    df = df.copy()
    df[LABEL_COL] = labels
    Z, _ = project_2d(X, use_umap=False)
    df["_z0"], df["_z1"] = Z[:, 0], Z[:, 1]

    lines: list[str] = []
    for cid in sorted(df[LABEL_COL].unique()):
        sub = df[df[LABEL_COL] == cid].copy()
        sub["_dist"] = np.hypot(sub["_z0"] - sub["_z0"].mean(), sub["_z1"] - sub["_z1"].mean())
        lines.append(f"# Cluster {cid} (size={len(sub)})")
        for desc in sub.nsmallest(samples_per_cluster, "_dist")[description_col]:
            lines.append(f"- {str(desc)[:200].replace(chr(10), ' ')}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote cluster report to: %s", report_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cluster bank transactions by description using embeddings + KMeans."
    )
    p.add_argument("--csv_path", required=True, help="Path to input CSV with transactions.")
    p.add_argument("--description_col", default="description", help="Description column name.")
    p.add_argument(
        "--embedding_provider", choices=["openai", "local"], default="openai", help="Embedding backend."
    )
    p.add_argument("--openai_model", default="text-embedding-3-small", help="OpenAI model.")
    p.add_argument("--local_model", default="all-MiniLM-L6-v2", help="sentence-transformers model.")
    p.add_argument("--device", default="cpu", help="Device for local model: cpu or cuda.")
    p.add_argument("--batch_size", type=int, default=100, help="Batch size for embedding calls.")
    p.add_argument("--n_clusters", type=int, default=10, help="Number of clusters if --auto_k off.")
    p.add_argument("--auto_k", action="store_true", help="Search k in [k_min, k_max] via silhouette.")
    p.add_argument("--k_min", type=int, default=8, help="Min k for auto search (>=2).")
    p.add_argument("--k_max", type=int, default=20, help="Max k for auto search.")
    p.add_argument("--plot_path", default=None, help="If set, write a 2D plot (PCA or UMAP).")
    p.add_argument("--use_umap", action="store_true", help="Use UMAP for the plot if available.")
    p.add_argument("--report_path", default=None, help="If set, write a cluster report.")
    p.add_argument("--samples_per_cluster", type=int, default=5, help="Samples per cluster in report.")
    p.add_argument("--out_path", default="./transactions_labeled.csv", help="Output labeled CSV.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Loading data...")
    df = load_dataframe(args.csv_path, args.description_col)

    logger.info(
        "Embedding %d descriptions using provider '%s'...", len(df), args.embedding_provider
    )
    X = embed_texts(df[CLEAN_COL].tolist(), args)

    if args.auto_k:
        logger.info(
            "Selecting best k in [%d, %d] via silhouette score...", args.k_min, args.k_max
        )
        k, _ = choose_best_k(X, k_min=args.k_min, k_max=args.k_max)
    else:
        k = args.n_clusters
        logger.info("Using specified k=%d", k)

    logger.info("Clustering into %d clusters...", k)
    labels, _ = fit_kmeans(X, k)
    df[LABEL_COL] = labels

    logger.info("Cluster sizes:")
    for cid, cnt in df[LABEL_COL].value_counts().sort_index().items():
        logger.info("  Cluster %s: %d", cid, cnt)

    if args.plot_path:
        try:
            plot_2d(X, labels, path=args.plot_path, use_umap=args.use_umap)
        except Exception as exc:
            logger.error("Plotting failed: %s", exc)

    if args.report_path:
        try:
            write_cluster_report(
                df, labels, X, args.description_col, args.report_path, args.samples_per_cluster
            )
        except Exception as exc:
            logger.error("Report generation failed: %s", exc)

    logger.info("Writing labeled data to: %s", args.out_path)
    df.drop(columns=[CLEAN_COL], errors="ignore").to_csv(args.out_path, index=False)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
