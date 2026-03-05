"""
Verify NaN nodata merge logic in reproject_datatree.

Demonstrates:
1. The bug: `combined[var] == nan` is always False, so merge silently fails
2. The fix: NaN-aware logic uses `.isnull()` instead
3. Regression check: int8 with nodata=-128 still works
4. Dataset-level xr.where works when all args are Datasets
"""

import math

import numpy as np
import xarray as xr


def make_ds(data, nodata, dtype="float32"):
    """Create a small Dataset with a single 'embeddings' variable."""
    arr = np.array(data, dtype=dtype)
    return xr.Dataset(
        {
            "embeddings": xr.DataArray(
                arr, dims=["band", "y", "x"], attrs={"nodata": nodata}
            )
        }
    )


def merge_buggy(combined, ds):
    """Old logic: `combined[var] == nodata` — broken for NaN."""
    for var in combined.data_vars:
        if var not in ds.data_vars:
            continue
        nodata = combined[var].attrs.get("nodata")
        if nodata is not None:
            mask = combined[var] == nodata
        else:
            mask = combined[var].isnull()
        combined[var] = xr.where(mask, ds[var], combined[var], keep_attrs=True)
    return combined


def merge_fixed(combined, ds):
    """Fixed logic: use `.isnull()` when nodata is NaN."""
    for var in combined.data_vars:
        if var not in ds.data_vars:
            continue
        nodata = combined[var].attrs.get("nodata")
        if nodata is not None and not (
            isinstance(nodata, float) and math.isnan(nodata)
        ):
            mask = combined[var] == nodata
        else:
            mask = combined[var].isnull()
        combined[var] = xr.where(mask, ds[var], combined[var], keep_attrs=True)
    return combined


def test_nan_nodata_bug():
    """Show that == NaN always gives False, so no merge happens."""
    print("=== Test 1: NaN nodata bug (old logic) ===")
    # ds_a has data in pixel [0,0,0], NaN in [0,0,1]
    # ds_b has NaN in pixel [0,0,0], data in [0,0,1]
    ds_a = make_ds([[[1.0, np.nan]]], nodata=np.nan)
    ds_b = make_ds([[[np.nan, 2.0]]], nodata=np.nan)

    result = merge_buggy(ds_a.copy(deep=True), ds_b)
    vals = result["embeddings"].values
    # With the bug, pixel [0,0,1] stays NaN because `nan == nan` is False
    assert np.isnan(vals[0, 0, 1]), "Bug demo: pixel should still be NaN (merge failed)"
    print("  PASS: Confirmed bug — NaN pixel was NOT filled\n")


def test_nan_nodata_fix():
    """Show that the fixed logic correctly merges NaN nodata."""
    print("=== Test 2: NaN nodata fix (new logic) ===")
    ds_a = make_ds([[[1.0, np.nan]]], nodata=np.nan)
    ds_b = make_ds([[[np.nan, 2.0]]], nodata=np.nan)

    result = merge_fixed(ds_a.copy(deep=True), ds_b)
    vals = result["embeddings"].values
    assert vals[0, 0, 0] == 1.0, "Pixel from ds_a should be preserved"
    assert vals[0, 0, 1] == 2.0, "NaN pixel should be filled from ds_b"
    print("  PASS: NaN pixels correctly merged\n")


def test_int8_regression():
    """Ensure int8 nodata=-128 still works with the fixed logic."""
    print("=== Test 3: int8 nodata=-128 regression check ===")
    ds_a = make_ds([[[10, -128]]], nodata=-128, dtype="int8")
    ds_b = make_ds([[[-128, 20]]], nodata=-128, dtype="int8")

    result = merge_fixed(ds_a.copy(deep=True), ds_b)
    vals = result["embeddings"].values
    assert vals[0, 0, 0] == 10, "Pixel from ds_a should be preserved"
    assert vals[0, 0, 1] == 20, "Nodata pixel should be filled from ds_b"
    print("  PASS: int8 merge works correctly\n")


def test_dataset_level_where():
    """Confirm xr.where works with Dataset-level arguments."""
    print("=== Test 4: Dataset-level xr.where ===")
    ds_a = make_ds([[[1.0, np.nan]]], nodata=np.nan)
    ds_b = make_ds([[[np.nan, 2.0]]], nodata=np.nan)

    mask_ds = ds_a.isnull()
    result = xr.where(mask_ds, ds_b, ds_a, keep_attrs=True)
    vals = result["embeddings"].values
    assert vals[0, 0, 0] == 1.0
    assert vals[0, 0, 1] == 2.0
    print("  PASS: Dataset-level xr.where works\n")


if __name__ == "__main__":
    test_nan_nodata_bug()
    test_nan_nodata_fix()
    test_int8_regression()
    test_dataset_level_where()
    print("All tests passed!")
