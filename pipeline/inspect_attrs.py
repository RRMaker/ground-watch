import glob, h5py, numpy as np

path = glob.glob("data/raw/frisco/*.nc")[0]
with h5py.File(path, "r") as f:
    for name in ["spatial_ref", "displacement", "recommended_mask"]:
        print("\n", name, dict(f[name].attrs))
    for name in ["reference_datetime", "secondary_datetime"]:
        print(name, f["identification/" + name][()].decode())
    d = f["displacement"][::20, ::20]
    m = f["recommended_mask"][::20, ::20]
    print("displacement min/max, fraction NaN:", np.nanmin(d), np.nanmax(d), np.isnan(d).mean())
    print("mask values:", np.unique(m, return_counts=True))