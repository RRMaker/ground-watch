import glob, h5py

path = glob.glob("data/raw/frisco/*.nc")[0]
with h5py.File(path, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)
    f.visititems(show)