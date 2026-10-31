import argparse
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import earthaccess
import h5py
import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
 
regions = {  
    "frisco": (-96.95, 33.0, -96.65, 33.3),
    "houston": (-95.8, 29.4, -95.0, 30.1),
}
 
 
def pixel_window(f, bbox):
    sr = f["spatial_ref"].attrs
    wkt = sr["crs_wkt"]
    wkt = wkt.decode() if isinstance(wkt, bytes) else wkt
    gt = sr["GeoTransform"]
    gt = gt.decode() if isinstance(gt, bytes) else gt
    x0, dx, _, y0, _, dy = map(float, gt.split()) 
    left, bottom, right, top = transform_bounds("EPSG:4326", CRS.from_wkt(wkt), *bbox)
    nrows, ncols = f["displacement"].shape
    c0 = max(int(np.floor((left - x0) / dx)), 0)
    c1 = min(int(np.ceil((right - x0) / dx)), ncols)
    r0 = max(int(np.floor((top - y0) / dy)), 0)
    r1 = min(int(np.ceil((bottom - y0) / dy)), nrows)
    transform = (x0 + c0 * dx, dx, 0.0, y0 + r0 * dy, 0.0, dy)  
    return r0, r1, c0, c1, transform, wkt
 
 
def crop_one(f, bbox):
    r0, r1, c0, c1, transform, wkt = pixel_window(f, bbox)
    if r1 <= r0 or c1 <= c0:
        return None
    return dict(
        displacement=f["displacement"][r0:r1, c0:c1].astype("float32"),
        mask=f["recommended_mask"][r0:r1, c0:c1],
        transform=np.array(transform),
        crs_wkt=wkt,
        reference=f["identification/reference_datetime"][()].decode(),
        secondary=f["identification/secondary_datetime"][()].decode(),
    )
 
 
def out_name(granule_ur):
    parts = granule_ur.split("_")
    return f"{parts[4]}_{parts[6][:8]}_{parts[7][:8]}.npz" 
 
 
def thin(todo, every):
    if every <= 1:
        return todo
    groups = defaultdict(list)
    for r, name in todo:
        frame, ref, sec = name[:-4].split("_")
        groups[(frame, ref)].append((sec, r, name))
    kept = []
    for items in groups.values():
        items.sort(key=lambda x: x[0])
        for i, (_, r, name) in enumerate(items):
            if i % every == 0 or i == len(items) - 1:
                kept.append((r, name))
    return kept
 
 
def save(out, name, c, t):
    if c is None:
        print(name, "does not overlap the region, skipped")
        return
    np.savez_compressed(out / name, **c)
    print(f"{name}: {c['displacement'].shape} in {time.time() - t:.1f}s")
 
 
def crop_remote(r, bbox):
    fobj = earthaccess.open([r])[0]
    with h5py.File(fobj, "r") as f:
        return crop_one(f, bbox)
 
 
stall_seconds = 90       
max_file_seconds = 1800  
tries = 3
 
 
def nc_url(r):
    links = [u for u in r.data_links() if u.endswith(".nc")]
    want = r["umm"]["GranuleUR"] + ".nc"
    exact = [u for u in links if u.endswith("/" + want)]
    return (exact or links)[0]
 
 
def fetch(url, dest):
    for attempt in range(1, tries + 1):
        part = dest.with_name(dest.name + ".part")
        try:
            session = earthaccess.get_requests_https_session()
            start = time.time()
            with session.get(url, stream=True, timeout=(30, stall_seconds)) as resp:
                resp.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
                        if time.time() - start > max_file_seconds:
                            raise TimeoutError("file took over 30 min")
            part.rename(dest)
            return True
        except Exception as e:
            part.unlink(missing_ok=True)
            print(f"  {dest.name[:60]}...: {type(e).__name__}, attempt {attempt} of {tries}")
            time.sleep(10)
    return False
 
 
def crop_downloaded(batch, bbox, tmp, out):
    t = time.time()
    jobs = [(r, name, tmp / (r["umm"]["GranuleUR"] + ".nc")) for r, name in batch]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:  
        ok = list(pool.map(lambda j: fetch(nc_url(j[0]), j[2]), jobs))
    for (r, name, nc), good in zip(jobs, ok):
        if not good:
            print(name, "download failed 3 times, skipped (re-run the script to retry)")
            continue
        with h5py.File(nc, "r") as f:
            save(out, name, crop_one(f, bbox), t)
        nc.unlink()  
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region", choices=regions)
    ap.add_argument("--limit", type=int, help="only fetch N granules (speed test)")
    ap.add_argument("--every", type=int, default=1,
                    help="keep every Nth date per reference group (2 = half the files)")
    ap.add_argument("--download", action="store_true", help="skip the remote-read attempt")
    ap.add_argument("--batch", type=int, default=4, help="files per download batch")
    ap.add_argument("--frame", help="only this frame, e.g. F08882")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-10-31")
    args = ap.parse_args()
 
    bbox = regions[args.region]
    out = Path("data/crops") / args.region
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path("data/raw") / args.region / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)  
    tmp.mkdir(parents=True)
 
    earthaccess.login()
    results = earthaccess.search_data(
        short_name="OPERA_L3_DISP-S1_V1",
        bounding_box=bbox,
        temporal=(args.start, args.end),
        count=5000, 
    )
    print(len(results), "granules found")
    todo = [(r, out_name(r["umm"]["GranuleUR"])) for r in results]
    if args.frame:
        todo = [(r, n) for r, n in todo if n.startswith(args.frame + "_")]
        print(len(todo), "in frame", args.frame)
    todo = thin(todo, args.every)
    todo = [(r, n) for r, n in todo if not (out / n).exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(len(todo), "to fetch")
    if not todo:
        return
 
    if not args.download:
        r, name = todo[0]
        t = time.time()
        try:
            save(out, name, crop_remote(r, bbox), t)
            remote_ok = True
        except Exception as e:
            print(f"Remote read failed ({type(e).__name__}), switching to download-crop-delete.")
            remote_ok = False
        if remote_ok:
            for r, name in todo[1:]:
                t = time.time()
                save(out, name, crop_remote(r, bbox), t)
            return
 
    for i in range(0, len(todo), args.batch):
        print(f"batch {i // args.batch + 1} of {-(-len(todo) // args.batch)}")
        crop_downloaded(todo[i : i + args.batch], bbox, tmp, out)
    shutil.rmtree(tmp, ignore_errors=True)
 
 
if __name__ == "__main__":
    main()
 