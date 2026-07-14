from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re
import zipfile

import numpy as np
import pandas as pd

REQUIRED_BASIN_MONTH_COLUMNS = {"date", "basin_id", "twsa_cm"}


def print_basin_month_requirements() -> None:
    print(
        "Missing basin-month GRACE data.\n"
        "Provide data/processed/basin_month_grace.csv with columns:\n"
        "  date, basin_id, twsa_cm, optional basin_name\n"
        "or place GRACE NetCDF and HydroBASINS polygons in data/raw/ and run notebook 01."
    )


def read_basin_month_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print_basin_month_requirements()
        return pd.DataFrame()
    df = pd.read_csv(path)
    missing = REQUIRED_BASIN_MONTH_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["basin_id"] = df["basin_id"].astype(str)
    if "basin_name" not in df.columns:
        df["basin_name"] = df["basin_id"]
    df = df.sort_values(["basin_id", "date"]).reset_index(drop=True)
    return df


def find_first_file(directory: Path, suffixes: Iterable[str]) -> Path | None:
    suffixes = tuple(s.lower() for s in suffixes)
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            return path
    return None


def find_grace_netcdf(directory: Path, expected_name: str | None = None) -> Path:
    """Find the GRACE mascon NetCDF without falling through to unrelated .nc files."""
    candidates = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".nc", ".nc4"}
        and ("grace" in path.name.lower() or "grctellus" in path.name.lower())
    )
    if expected_name is not None:
        exact = [path for path in candidates if path.name == expected_name]
        if exact:
            return exact[0]
        expected_path = directory / expected_name
        raise FileNotFoundError(f"Expected GRACE NetCDF was not found: {expected_path}")
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No GRACE NetCDF found in {directory}")
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"Multiple GRACE NetCDF candidates found; specify one explicitly: {names}")


def find_mask_zips(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.zip"))


def parse_mask_name(name: str, strict: bool = False) -> tuple[str, str]:
    stem = Path(name).name
    for suffix in [".mask.xyz", ".mask.csv"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    match = re.match(r"HyBas_(\d+)_(.+?)_Lev\d+_quartdeg", stem)
    if not match:
        if strict:
            raise ValueError(f"Could not parse HydroBASINS mask filename: {name}")
        return stem, stem
    return match.group(1), match.group(2).replace("_", " ")


def list_mask_members(
    mask_zips: list[Path],
    name_filter: str | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    rows = []
    for zip_path in mask_zips:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if not (member.endswith(".mask.xyz") or member.endswith(".mask.csv")):
                    continue
                basin_id, basin_name = parse_mask_name(member, strict=strict)
                if name_filter and name_filter.lower() not in basin_name.lower():
                    continue
                rows.append(
                    {
                        "zip_path": str(zip_path),
                        "member_name": member,
                        "basin_id": basin_id,
                        "basin_name": basin_name,
                    }
                )
    return pd.DataFrame(rows)


def read_positive_mask_cells_from_zip(zip_path: Path, member_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as handle:
            if member_name.endswith(".mask.csv"):
                df = pd.read_csv(handle, names=["lon", "lat", "weight"])
            else:
                df = pd.read_csv(
                    handle,
                    sep=r"\s+",
                    names=["lon", "lat", "weight"],
                    engine="python",
                )
    df = df[df["weight"] > 0].copy()
    df["lon"] = df["lon"].astype(float)
    df["lat"] = df["lat"].astype(float)
    df["weight"] = df["weight"].astype(float)
    return df


def download_grace_from_podaac(
    output_dir: Path,
    short_name: str = "TELLUS_GRAC-GRFO_MASCON_GRID_RL06.3_V4",
) -> list[Path]:
    """Download the GRACE mascon NetCDF from PO.DAAC using NASA Earthdata auth."""
    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError(
            "Install earthaccess to download from PO.DAAC: pip install earthaccess"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Authenticating with NASA Earthdata through earthaccess.")
    try:
        earthaccess.login(strategy="environment", persist=False)
    except Exception:
        try:
            earthaccess.login(strategy="netrc")
        except Exception:
            earthaccess.login(strategy="interactive", persist=True)
    print(f"Searching PO.DAAC collection: {short_name}")
    results = earthaccess.search_data(
        short_name=short_name,
        cloud_hosted=True,
    )
    if not results:
        raise FileNotFoundError(f"No PO.DAAC granules found for {short_name}.")
    print(f"Found {len(results)} granule(s). Downloading to {output_dir}.")
    downloaded = earthaccess.download(results, local_path=str(output_dir))
    return [Path(p) for p in downloaded]


def load_basin_polygons(path: Path):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Install geopandas to read basin polygons.") from exc
    basins = gpd.read_file(path)
    id_candidates = ["HYBAS_ID", "hybas_id", "basin_id", "BASIN_ID"]
    id_col = next((col for col in id_candidates if col in basins.columns), None)
    if id_col is None:
        raise ValueError(
            "Basin polygons need an ID column such as HYBAS_ID or basin_id."
        )
    basins = basins.rename(columns={id_col: "basin_id"}).copy()
    basins["basin_id"] = basins["basin_id"].astype(str)
    if "basin_name" not in basins.columns:
        basins["basin_name"] = basins["basin_id"]
    return basins


def filter_amazon_basins(basins, amazon_main_bas_id: str | None = None):
    """Filter a HydroBASINS layer to the Amazon region when metadata allows it."""
    if amazon_main_bas_id is not None:
        candidate_cols = [c for c in ["MAIN_BAS", "main_bas", "PFAF_ID", "pfaf_id"] if c in basins.columns]
        if not candidate_cols:
            raise ValueError("No MAIN_BAS/PFAF_ID-style column found for Amazon filtering.")
        col = candidate_cols[0]
        return basins[basins[col].astype(str).str.startswith(str(amazon_main_bas_id))].copy()

    if "SUB_AREA" in basins.columns:
        # For HydroBASINS inputs, this keeps a manageable first-pass subset.
        # Users can override with amazon_main_bas_id for a precise boundary.
        large = basins.sort_values("SUB_AREA", ascending=False).head(1)
        main_cols = [c for c in ["MAIN_BAS", "main_bas"] if c in basins.columns]
        if main_cols:
            main_id = str(large.iloc[0][main_cols[0]])
            return basins[basins[main_cols[0]].astype(str) == main_id].copy()
    print(
        "Could not infer an Amazon subset from polygon metadata. "
        "Using all supplied basins. Set amazon_main_bas_id in notebook 01 for a precise subset."
    )
    return basins.copy()


def aggregate_grace_netcdf_to_basins(
    grace_nc_path: Path,
    basin_path: Path,
    output_csv: Path,
    amazon_main_bas_id: str | None = None,
) -> pd.DataFrame:
    """Area-average GRACE grid cells whose centers fall inside each basin polygon."""
    try:
        import geopandas as gpd
        import xarray as xr
        from shapely.geometry import Point
    except ImportError as exc:
        raise ImportError("Install xarray, netcdf4, geopandas, and shapely for raw preprocessing.") from exc

    basins = filter_amazon_basins(load_basin_polygons(basin_path), amazon_main_bas_id)
    if basins.empty:
        raise ValueError("No basins found after Amazon filtering.")
    if basins.crs is None:
        basins = basins.set_crs("EPSG:4326")
    basins = basins.to_crs("EPSG:4326")

    ds = xr.open_dataset(grace_nc_path)
    var_candidates = ["lwe_thickness", "twsa", "TWSA", "water_thickness", "cmwe"]
    var_name = next((v for v in var_candidates if v in ds.data_vars), None)
    if var_name is None:
        numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        if not numeric_vars:
            raise ValueError("No numeric GRACE data variable found in NetCDF.")
        var_name = numeric_vars[0]

    lat_name = next((n for n in ["lat", "latitude", "y"] if n in ds.coords), None)
    lon_name = next((n for n in ["lon", "longitude", "x"] if n in ds.coords), None)
    time_name = next((n for n in ["time", "date"] if n in ds.coords), None)
    if not all([lat_name, lon_name, time_name]):
        raise ValueError("NetCDF needs recognizable time, latitude, and longitude coordinates.")

    da = ds[var_name]
    lon = ds[lon_name].values
    lat = ds[lat_name].values
    lon_for_geometry = ((lon + 180) % 360) - 180 if np.nanmax(lon) > 180 else lon
    lon_grid, lat_grid = np.meshgrid(lon_for_geometry, lat)
    points = gpd.GeoDataFrame(
        {"grid_id": np.arange(lon_grid.size)},
        geometry=[Point(x, y) for x, y in zip(lon_grid.ravel(), lat_grid.ravel())],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, basins[["basin_id", "basin_name", "geometry"]], predicate="within", how="inner")
    if joined.empty:
        raise ValueError("No GRACE grid-cell centers fell inside the supplied basin polygons.")

    values = da.transpose(time_name, lat_name, lon_name).values.reshape(len(ds[time_name]), -1)
    weights = np.cos(np.deg2rad(lat_grid)).ravel()
    rows = []
    for basin_id, group in joined.groupby("basin_id"):
        idx = group["grid_id"].to_numpy()
        basin_values = values[:, idx]
        basin_weights = weights[idx]
        valid = np.isfinite(basin_values)
        weighted = np.where(valid, basin_values, 0.0) * basin_weights
        denom = valid @ basin_weights
        series = np.divide(weighted.sum(axis=1), denom, out=np.full(len(ds[time_name]), np.nan), where=denom > 0)
        basin_name = group["basin_name"].iloc[0]
        rows.append(pd.DataFrame({
            "date": pd.to_datetime(ds[time_name].values),
            "basin_id": str(basin_id),
            "basin_name": basin_name,
            "twsa_cm": series,
        }))
    out = pd.concat(rows, ignore_index=True).sort_values(["basin_id", "date"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def _nearest_indices(source_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    source = np.asarray(source_values, dtype=float)
    target = np.asarray(target_values, dtype=float)
    order = np.argsort(source)
    sorted_source = source[order]
    pos = np.searchsorted(sorted_source, target)
    pos = np.clip(pos, 1, len(sorted_source) - 1)
    left = sorted_source[pos - 1]
    right = sorted_source[pos]
    choose_right = np.abs(right - target) < np.abs(target - left)
    nearest_sorted = np.where(choose_right, pos, pos - 1)
    return order[nearest_sorted]


def aggregate_grace_netcdf_to_mask_zips(
    grace_nc_path: Path,
    mask_zips: list[Path],
    output_csv: Path,
    basin_name_filter: str | None = None,
    basin_name_exclude: str | None = None,
) -> pd.DataFrame:
    """Aggregate GRACE NetCDF to HydroBASINS .mask.xyz files stored in zip archives."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("Install xarray and netcdf4 for GRACE NetCDF preprocessing.") from exc

    members = list_mask_members(mask_zips, basin_name_filter, strict=True)
    if basin_name_exclude and not members.empty:
        members = members[
            ~members["basin_name"].str.contains(basin_name_exclude, case=False, na=False)
        ].copy()
    if members.empty:
        raise FileNotFoundError("No .mask.xyz files matched the requested basin filter.")
    from .validation import validate_unique_mask_members

    validate_unique_mask_members(members)

    ds = xr.open_dataset(grace_nc_path)
    var_candidates = ["lwe_thickness", "twsa", "TWSA", "water_thickness", "cmwe"]
    var_name = next((v for v in var_candidates if v in ds.data_vars), None)
    if var_name is None:
        numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        if not numeric_vars:
            raise ValueError("No numeric GRACE data variable found in NetCDF.")
        var_name = numeric_vars[0]

    lat_name = next((n for n in ["lat", "latitude", "y"] if n in ds.coords), None)
    lon_name = next((n for n in ["lon", "longitude", "x"] if n in ds.coords), None)
    time_name = next((n for n in ["time", "date"] if n in ds.coords), None)
    if not all([lat_name, lon_name, time_name]):
        raise ValueError("NetCDF needs recognizable time, latitude, and longitude coordinates.")

    da = ds[var_name].transpose(time_name, lat_name, lon_name)
    values = da.values
    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)
    lon_geometry = ((lon + 180) % 360) - 180 if np.nanmax(lon) > 180 else lon
    dates = pd.to_datetime(ds[time_name].values)
    rows = []

    for item in members.itertuples(index=False):
        print(f"Aggregating {item.basin_name}")
        mask = read_positive_mask_cells_from_zip(Path(item.zip_path), item.member_name)
        mask_lon = ((mask["lon"].to_numpy() + 180) % 360) - 180
        mask_lat = mask["lat"].to_numpy()
        lat_idx = _nearest_indices(lat, mask_lat)
        lon_idx = _nearest_indices(lon_geometry, mask_lon)
        cell_values = values[:, lat_idx, lon_idx]
        weights = mask["weight"].to_numpy() * np.cos(np.deg2rad(mask_lat))
        valid = np.isfinite(cell_values)
        weighted = np.where(valid, cell_values, 0.0) * weights
        denom = valid @ weights
        series = np.divide(weighted.sum(axis=1), denom, out=np.full(len(dates), np.nan), where=denom > 0)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "basin_id": str(item.basin_id),
                    "basin_name": item.basin_name,
                    "twsa_cm": series,
                }
            )
        )

    out = pd.concat(rows, ignore_index=True).sort_values(["basin_id", "date"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def aggregate_era5_netcdf_to_mask_zips(
    era5_nc_path: Path,
    mask_zips: list[Path],
    output_csv: Path,
    basin_name_filter: str | None = None,
    basin_name_exclude: str | None = None,
) -> pd.DataFrame:
    """Aggregate ERA5 monthly fields to HydroBASINS masks with mask and area weights.

    ERA5 total precipitation, evaporation, and runoff are read in meters and written
    in millimeters. Evaporation is stored as positive loss magnitude.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("Install xarray and netcdf4 for ERA5 NetCDF preprocessing.") from exc

    members = list_mask_members(mask_zips, basin_name_filter, strict=True)
    if basin_name_exclude and not members.empty:
        members = members[
            ~members["basin_name"].str.contains(basin_name_exclude, case=False, na=False)
        ].copy()
    if members.empty:
        raise FileNotFoundError("No .mask.xyz files matched the requested basin filter.")
    from .validation import validate_unique_mask_members

    validate_unique_mask_members(members)

    ds = xr.open_dataset(era5_nc_path)
    required_vars = {"tp", "e", "ro"}
    missing_vars = required_vars - set(ds.data_vars)
    if missing_vars:
        raise ValueError(f"ERA5 NetCDF is missing variables: {sorted(missing_vars)}")

    lat_name = next((n for n in ["lat", "latitude", "y"] if n in ds.coords), None)
    lon_name = next((n for n in ["lon", "longitude", "x"] if n in ds.coords), None)
    time_name = next((n for n in ["valid_time", "time", "date"] if n in ds.coords), None)
    if not all([lat_name, lon_name, time_name]):
        raise ValueError("ERA5 NetCDF needs recognizable time, latitude, and longitude coordinates.")

    variables = {
        "era5_tp_mm": ds["tp"].transpose(time_name, lat_name, lon_name).values * 1000.0,
        "era5_evap_mm": ds["e"].transpose(time_name, lat_name, lon_name).values * -1000.0,
        "era5_ro_mm": ds["ro"].transpose(time_name, lat_name, lon_name).values * 1000.0,
    }
    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)
    lon_geometry = ((lon + 180) % 360) - 180 if np.nanmax(lon) > 180 else lon
    dates = pd.to_datetime(ds[time_name].values).to_period("M").to_timestamp()
    rows = []

    for item in members.itertuples(index=False):
        print(f"Aggregating ERA5 for {item.basin_name}")
        mask = read_positive_mask_cells_from_zip(Path(item.zip_path), item.member_name)
        mask_lon = ((mask["lon"].to_numpy() + 180) % 360) - 180
        mask_lat = mask["lat"].to_numpy()
        lat_idx = _nearest_indices(lat, mask_lat)
        lon_idx = _nearest_indices(lon_geometry, mask_lon)
        weights = mask["weight"].to_numpy() * np.cos(np.deg2rad(mask_lat))
        out = {
            "date": dates,
            "basin_id": str(item.basin_id),
            "basin_name": item.basin_name,
        }
        for column, values in variables.items():
            cell_values = values[:, lat_idx, lon_idx]
            valid = np.isfinite(cell_values)
            weighted = np.where(valid, cell_values, 0.0) * weights
            denom = valid @ weights
            out[column] = np.divide(
                weighted.sum(axis=1),
                denom,
                out=np.full(len(dates), np.nan),
                where=denom > 0,
            )
        rows.append(pd.DataFrame(out))

    out_df = pd.concat(rows, ignore_index=True).sort_values(["basin_id", "date"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df
