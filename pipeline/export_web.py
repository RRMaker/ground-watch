import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

regions = {
    "frisco": dict(frame="F08885", label="Frisco, TX"),
    "houston": dict(frame="F08882", label="Houston, TX"),
}
versions = {"plain": " ", "tilt": "_deramp"}
limit = 10

a_ = 6378137.0
f_ = 1 / 298.257223563
k0 = 0.9996
n_ = f_ / (2 - f_)
aa = a_ / (1 + n_) * (1 + n_**2 / 4 + n_**4 / 64)
e_ = 2 * np.sqrt(n_) / (1 + n_)  
ALPHA = [n_ / 2 - 2 * n_**2 / 3 + 5 * n_**3 / 16, 13 * n_**2 / 48 - 3 * n_**3 / 5, 61 * n_**3 / 240]
BETA = [n_ / 2 - 2 * n_**2 / 3 + 37 * n_**3 / 96, n_**2 / 48 + n_**3 / 15, 17 * n_**3 / 480]
DELTA = [2 * n_ - 2 * n_**2 / 3 - 2 * n_**3, 7 * n_**2 / 3 - 8 * n_**3 / 5, 56 * n_**3 / 15]

def utm_zone(wkt):
    m = re.search(r"UTM zone (\d+)([NS])", wkt)
    if not m:
        raise ValueError("could not find a UTM zone in the file's projection info")
    return int(m.group(1)), m.group(2) == "N"

def ll_to_utm(lat, lon, zone, north=True):
    phi, lam = np.radians(lat), np.radians(lon - (zone * 6 - 183))
    t = np.sinh(np.arctanh(np.sin(phi)) - e_ * np.arctanh(e_ * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    x, y = eta.copy(), xi.copy()
    for j, a in enumerate(ALPHA, 1):
        x += a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
        y += a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
    return 500000 + k0 * aa * x, (0 if north else 10000000) + k0 * aa * y


def utm_to_ll(e, n, zone, north=True):
    xi = (n - (0 if north else 10000000)) / (k0 * aa)
    eta = (e - 500000) / (k0 * aa)
    xp, ep = xi.copy(), eta.copy()
    for j, b in enumerate(BETA, 1):
        xp -= b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        ep -= b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xp) / np.cosh(ep))
    phi = chi.copy()
    for j, d in enumerate(DELTA, 1):
        phi += d * np.sin(2 * j * chi)
    lam = np.arctan2(np.sinh(ep), np.cos(xp))
    return np.degrees(phi), np.degrees(lam) + (zone * 6 - 183)


r = 6378137.0


def merc(lat, lon):
    return r * np.radians(lon), r * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def unmerc(x, y):
    return np.degrees(2 * np.arctan(np.exp(y / r)) - np.pi / 2), np.degrees(x / r)

def to_web_image(rate, tf, zone, north):
    ny, nx = rate.shape
    x0, dx, _, y0, _, dy = tf
    s = np.linspace(0, 1, 50)
    edge_c = np.concatenate([s * nx, np.full(50, nx), s * nx, np.zeros(50)])
    edge_r = np.concatenate([np.zeros(50), s * ny, np.full(50, ny), s * ny])
    lat, lon = utm_to_ll(x0 + edge_c * dx, y0 + edge_r * dy, zone, north)
    mx, my = merc(lat, lon)
    lat_c = np.radians(lat.mean())
    res = abs(dx) / np.cos(lat_c) 
    w = int(np.ceil((mx.max() - mx.min()) / res))
    h = int(np.ceil((my.max() - my.min()) / res))
    gx = mx.min() + (np.arange(w) + 0.5) * res
    gy = my.max() - (np.arange(h) + 0.5) * res
    GX, GY = np.meshgrid(gx, gy)
    la, lo = unmerc(GX, GY)
    e, n = ll_to_utm(la, lo, zone, north)
    col = np.floor((e - x0) / dx).astype(int)
    row = np.floor((n - y0) / dy).astype(int)
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    out = np.full((h, w), np.nan, dtype="float32")
    out[inside] = rate[row[inside], col[inside]]
    rgba = plt.get_cmap("RdBu")(np.clip((out + limit) / (2 * limit), 0, 1))
    rgba[..., 3] = np.isfinite(out)
    south, west = unmerc(mx.min(), my.min())
    north_, east = unmerc(mx.max(), my.max())
    bounds = [[round(float(south), 6), round(float(west), 6)], [round(float(north_), 6), round(float(east), 6)]]
    return rgba, bounds
 
 
def main():
    outdir = Path("docs/data")
    outdir.mkdir(parents=True, exist_ok=True)
    meta = dict(limit=limit, regions={})
    for region, info in regions.items():
        prod = Path("data/products") / region
        entry = dict(label=info["label"], frame=info["frame"], versions={})
        for version, tag in versions.items():
            rate_f = prod / f"{info['frame']}{tag}_rate.npz"
            cell_f = prod / f"{info['frame']}{tag}_cells.npz"
            if not (rate_f.exists() and cell_f.exists()):
                print(f"{region} {version}: missing {rate_f.name} or {cell_f.name}, skipped "
                      "(re-run build_velocity.py for it)")
                continue
            r = np.load(rate_f)
            zone, north = utm_zone(str(r["crs_wkt"]))
            rgba, bounds = to_web_image(r["rate"], r["transform"], zone, north)
            png = f"{region}_{version}.png"
            plt.imsave(outdir / png, rgba)
 
            c = np.load(cell_f)
            lat, lon = utm_to_ll(c["x"], c["y"], zone, north)
            cells = [
                [round(float(a), 5), round(float(b), 5), int(k), round(float(m), 1),
                 round(float(q1), 1), round(float(q3), 1), [int(round(v)) for v in ser]]
                for a, b, k, m, q1, q3, ser in zip(lat, lon, c["n"], c["rate"], c["p25"], c["p75"], c["series"])
            ]
            dates = [str(d) for d in c["dates"]]
            js = f"{region}_{version}.json"
            (outdir / js).write_text(json.dumps(
                dict(dates=dates, cell_m=float(c["cell_m"]), px_m=abs(float(r["transform"][1])), cells=cells),
                separators=(",", ":")))
            entry["versions"][version] = dict(overlay=png, cells=js)
            entry.update(bounds=bounds, start=dates[0], end=dates[-1])
            print(f"{region} {version}: {rgba.shape[1]}x{rgba.shape[0]} map, {len(cells)} blocks, "
                  f"{(outdir / png).stat().st_size / 1e6:.1f} MB + {(outdir / js).stat().st_size / 1e6:.1f} MB")
        if entry["versions"]:
            meta["regions"][region] = entry
    (outdir / "regions.json").write_text(json.dumps(meta, indent=1))
    print("wrote", outdir)
 
 
if __name__ == "__main__":
    main()
 