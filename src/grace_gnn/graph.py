from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_edges_from_polygons(basins, output_csv: Path) -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Install geopandas to build polygon adjacency.") from exc
    if basins.crs is None:
        basins = basins.set_crs("EPSG:4326")
    left = basins[["basin_id", "geometry"]].copy()
    right = basins[["basin_id", "geometry"]].copy()
    joined = gpd.sjoin(left, right, predicate="intersects", how="inner", lsuffix="src", rsuffix="dst")
    src_col = "basin_id_src" if "basin_id_src" in joined.columns else "basin_id_left"
    dst_col = "basin_id_dst" if "basin_id_dst" in joined.columns else "basin_id_right"
    edges = joined[[src_col, dst_col]].rename(columns={src_col: "src_basin_id", dst_col: "dst_basin_id"})
    edges["src_basin_id"] = edges["src_basin_id"].astype(str)
    edges["dst_basin_id"] = edges["dst_basin_id"].astype(str)
    edges = edges[edges["src_basin_id"] != edges["dst_basin_id"]].drop_duplicates()
    edges["weight"] = 1.0
    edges["graph_type"] = "real"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output_csv, index=False)
    return edges


def load_edges(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(
            f"Missing edge file: {path}\n"
            "Expected columns: src_basin_id, dst_basin_id, optional weight, graph_type."
        )
        return pd.DataFrame()
    edges = pd.read_csv(path)
    required = {"src_basin_id", "dst_basin_id"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    edges["src_basin_id"] = edges["src_basin_id"].astype(str)
    edges["dst_basin_id"] = edges["dst_basin_id"].astype(str)
    if "weight" not in edges.columns:
        edges["weight"] = 1.0
    if "graph_type" not in edges.columns:
        edges["graph_type"] = "real"
    return edges


def make_random_edges(real_edges: pd.DataFrame, basin_ids: list[str], output_csv: Path, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_edges = len(real_edges)
    possible = [(a, b) for a in basin_ids for b in basin_ids if a != b]
    if n_edges > len(possible):
        raise ValueError("Requested more random edges than possible non-self edges.")
    choices = rng.choice(len(possible), size=n_edges, replace=False)
    edges = pd.DataFrame([possible[i] for i in choices], columns=["src_basin_id", "dst_basin_id"])
    edges["weight"] = 1.0
    edges["graph_type"] = "random"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output_csv, index=False)
    return edges


def subset_edges(edges: pd.DataFrame, basin_ids: list[str], graph_type: str | None = None) -> pd.DataFrame:
    keep = {str(bid) for bid in basin_ids}
    out = edges[
        edges["src_basin_id"].astype(str).isin(keep)
        & edges["dst_basin_id"].astype(str).isin(keep)
    ].copy()
    out["src_basin_id"] = out["src_basin_id"].astype(str)
    out["dst_basin_id"] = out["dst_basin_id"].astype(str)
    if graph_type is not None:
        out["graph_type"] = graph_type
    return out.reset_index(drop=True)


def symmetrize_edges(edges: pd.DataFrame, graph_type: str = "real_knn_undirected") -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    forward = edges[["src_basin_id", "dst_basin_id"]].copy()
    reverse = forward.rename(columns={"src_basin_id": "dst_basin_id", "dst_basin_id": "src_basin_id"})
    out = pd.concat([forward, reverse], ignore_index=True).drop_duplicates()
    out = out[out["src_basin_id"].astype(str) != out["dst_basin_id"].astype(str)]
    out["weight"] = 1.0
    out["graph_type"] = graph_type
    return out.reset_index(drop=True)


def reverse_edges(edges: pd.DataFrame, graph_type: str = "real_knn_reversed") -> pd.DataFrame:
    out = edges[["dst_basin_id", "src_basin_id"]].rename(
        columns={"dst_basin_id": "src_basin_id", "src_basin_id": "dst_basin_id"}
    )
    out = out.drop_duplicates()
    out["weight"] = 1.0
    out["graph_type"] = graph_type
    return out.reset_index(drop=True)


def make_degree_matched_random_edges(
    real_edges: pd.DataFrame,
    basin_ids: list[str],
    seed: int = 42,
    graph_type: str = "random_degree_matched",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    basin_ids = [str(bid) for bid in basin_ids]
    existing: set[tuple[str, str]] = set()
    rows = []
    out_degrees = real_edges.groupby(real_edges["src_basin_id"].astype(str)).size().to_dict()
    for src in basin_ids:
        degree = int(out_degrees.get(src, 0))
        candidates = [bid for bid in basin_ids if bid != src]
        if degree > len(candidates):
            raise ValueError(f"Cannot draw {degree} non-self random edges for node {src}.")
        rng.shuffle(candidates)
        picked = 0
        for dst in candidates:
            pair = (src, dst)
            if pair in existing:
                continue
            existing.add(pair)
            rows.append(pair)
            picked += 1
            if picked == degree:
                break
    edges = pd.DataFrame(rows, columns=["src_basin_id", "dst_basin_id"])
    edges["weight"] = 1.0
    edges["graph_type"] = graph_type
    return edges


def save_edges(edges: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output_csv, index=False)
    return edges


def _mask_centroid(mask: pd.DataFrame) -> np.ndarray | None:
    lon = np.deg2rad(((mask["lon"].to_numpy() + 180) % 360) - 180)
    lat = np.deg2rad(mask["lat"].to_numpy())
    weights = mask["weight"].to_numpy() * np.cos(lat)
    xyz = np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])
    center = np.average(xyz, axis=0, weights=weights)
    norm = np.linalg.norm(center)
    return center / norm if norm > 0 else None


def build_knn_edges_from_mask_zips(
    mask_zips: list[Path],
    basin_names: list[str],
    output_csv: Path,
    k: int = 3,
    graph_type: str = "real_knn_directed",
    strict_mask_names: bool = True,
) -> pd.DataFrame:
    from .data import list_mask_members, read_positive_mask_cells_from_zip

    keep = set(basin_names)
    members = list_mask_members(mask_zips, strict=strict_mask_names)
    members = members[members["basin_name"].isin(keep)].copy()
    if members.empty:
        raise FileNotFoundError("No mask files matched the requested basin names.")
    missing = sorted(keep - set(members["basin_name"]))
    if missing:
        raise FileNotFoundError(f"Missing mask files for basin names: {missing}")
    from .validation import validate_unique_mask_members

    validate_unique_mask_members(members)

    centroids = {}
    for item in members.itertuples(index=False):
        print(f"Reading regional kNN centroid cells for {item.basin_name}")
        mask = read_positive_mask_cells_from_zip(Path(item.zip_path), item.member_name)
        center = _mask_centroid(mask)
        if center is not None:
            centroids[str(item.basin_id)] = center
    basin_ids = sorted(centroids)
    if len(basin_ids) < 2:
        raise ValueError("Need at least two basin masks to build kNN edges.")

    vectors = np.vstack([centroids[bid] for bid in basin_ids])
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
    distances = np.arccos(similarities)
    edge_pairs = set()
    k = min(k, len(basin_ids) - 1)
    for i, src in enumerate(basin_ids):
        nearest = np.argsort(distances[i])[1 : k + 1]
        for j in nearest:
            edge_pairs.add((src, basin_ids[j]))
    edges = pd.DataFrame(sorted(edge_pairs), columns=["src_basin_id", "dst_basin_id"])
    edges["weight"] = 1.0
    edges["graph_type"] = graph_type
    return save_edges(edges, output_csv)


def normalized_adjacency(edges: pd.DataFrame, basin_ids: list[str], add_self_loops: bool = True):
    import torch

    idx = {bid: i for i, bid in enumerate(basin_ids)}
    n = len(basin_ids)
    adj = torch.zeros((n, n), dtype=torch.float32)
    for row in edges.itertuples(index=False):
        src = str(row.src_basin_id)
        dst = str(row.dst_basin_id)
        if src in idx and dst in idx:
            weight = float(getattr(row, "weight", 1.0))
            adj[idx[dst], idx[src]] = weight
    if add_self_loops:
        adj += torch.eye(n)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


def build_edges_from_mask_zips(
    mask_zips: list[Path],
    output_csv: Path,
    basin_name_filter: str | None = None,
    strict_mask_names: bool = True,
) -> pd.DataFrame:
    from .data import list_mask_members, read_positive_mask_cells_from_zip

    members = list_mask_members(mask_zips, basin_name_filter, strict=strict_mask_names)
    from .validation import validate_unique_mask_members

    validate_unique_mask_members(members)
    occupancy: dict[tuple[int, int], str] = {}
    centroids = {}
    for item in members.itertuples(index=False):
        print(f"Reading mask adjacency cells for {item.basin_name}")
        mask = read_positive_mask_cells_from_zip(Path(item.zip_path), item.member_name)
        lon_idx = np.rint(mask["lon"].to_numpy() * 4).astype(int)
        lat_idx = np.rint(mask["lat"].to_numpy() * 4).astype(int)
        lon = np.deg2rad(((mask["lon"].to_numpy() + 180) % 360) - 180)
        lat = np.deg2rad(mask["lat"].to_numpy())
        weights = mask["weight"].to_numpy() * np.cos(lat)
        xyz = np.column_stack([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])
        center = np.average(xyz, axis=0, weights=weights)
        norm = np.linalg.norm(center)
        if norm > 0:
            centroids[str(item.basin_id)] = center / norm
        for x, y in zip(lon_idx, lat_idx):
            occupancy[(int(x), int(y))] = str(item.basin_id)

    edge_pairs = set()
    for (x, y), basin_id in occupancy.items():
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            other = occupancy.get((x + dx, y + dy))
            if other and other != basin_id:
                edge_pairs.add((basin_id, other))

    graph_type = "real_mask_touching"
    if not edge_pairs and len(centroids) > 1:
        print("No touching mask edges found. Falling back to 3-nearest mask-centroid neighbors.")
        basin_ids = sorted(centroids)
        vectors = np.vstack([centroids[bid] for bid in basin_ids])
        similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
        distances = np.arccos(similarities)
        k = min(3, len(basin_ids) - 1)
        for i, src in enumerate(basin_ids):
            nearest = np.argsort(distances[i])[1 : k + 1]
            for j in nearest:
                edge_pairs.add((src, basin_ids[j]))
        graph_type = "real_mask_nearest"

    if edge_pairs:
        edges = pd.DataFrame(sorted(edge_pairs), columns=["src_basin_id", "dst_basin_id"])
        edges["weight"] = 1.0
        edges["graph_type"] = graph_type
    else:
        print("No mask-derived edges found.")
        edges = pd.DataFrame(columns=["src_basin_id", "dst_basin_id", "weight", "graph_type"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output_csv, index=False)
    return edges
