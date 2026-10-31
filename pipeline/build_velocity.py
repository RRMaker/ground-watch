import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
 
min_good_fraction = 0.7
 
 
def coarsen(disp, mask, n):
    if n == 1:
        return disp, mask == 1
    ny, nx = (disp.shape[0] // n) * n, (disp.shape[1] // n) * n
    d = disp[:ny, :nx].reshape(ny // n, n, nx // n, n)
    g = (mask[:ny, :nx] == 1).reshape(ny // n, n, nx // n, n) & np.isfinite(d)
    f = np.isfinite(d)
    with np.errstate(invalid="ignore", divide="ignore"):
        good_mean = np.where(g, d, 0).sum((1, 3)) / g.sum((1, 3))
        any_mean = np.where(f, d, 0).sum((1, 3)) / f.sum((1, 3))
    out = np.where(g.any((1, 3)), good_mean, any_mean)
    return out.astype("float32"), g.mean((1, 3)) >= 0.5
 
 
def load(region, n=1, frame="F"):
    frames = defaultdict(list)
    for p in sorted((Path("data/crops") / region).glob(f"{frame}*.npz")):
        z = np.load(p)
        disp, good = coarsen(z["displacement"].astype("float32") * 1000.0, z["mask"], n)  
        tf = z["transform"].copy()
        tf[1] *= n
        tf[5] *= n
        frames[p.stem.split("_")[0]].append(dict(
            ref=datetime.fromisoformat(str(z["reference"])),
            sec=datetime.fromisoformat(str(z["secondary"])),
            disp=disp,
            good=good,
            transform=tf,
            wkt=str(z["crs_wkt"]),
        ))
    return frames
 
 
def chain(items):
    items = sorted(items, key=lambda d: d["sec"])
    shape = items[0]["disp"].shape
    t0 = min(d["ref"] for d in items)
    refs = {d["ref"].date() for d in items}  
    cum_at = {t0.date(): np.zeros(shape, "float32")}
    dates, stack, masks = [t0], [np.zeros(shape, "float32")], [np.ones(shape, bool)]
    for d in items:
        if d["disp"].shape != shape:
            print("  skip (grid size differs):", d["sec"].date())
            continue
        base = cum_at.get(d["ref"].date())
        if base is None:
            print(f"  skip {d['sec'].date()}: no earlier file ends on its reference date {d['ref'].date()}")
            continue
        cum = base + d["disp"]
        if d["sec"].date() in refs:
            cum_at[d["sec"].date()] = cum
        dates.append(d["sec"])
        stack.append(cum)
        masks.append(d["good"])
    return dates, np.array(stack), np.array(masks)
 
 
def design(dates):
    t = np.array([(d - dates[0]).total_seconds() / (365.25 * 86400) for d in dates])
    return np.column_stack([np.ones_like(t), t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
 
 
def fit(dates, stack, good):
    A = design(dates)
    pinv = np.linalg.pinv(A)
    var_rate = np.linalg.inv(A.T @ A)[1, 1]
    dof = max(len(dates) - A.shape[1], 1)
    T, ny, nx = stack.shape
    flat = stack.reshape(T, -1)
    idx = np.flatnonzero(good.ravel())
    out = np.full((3, ny * nx), np.nan, dtype="float32")
    for i in range(0, len(idx), 200_000):  
        cols = idx[i : i + 200_000]
        Y = flat[:, cols].astype("float64")
        coef = pinv @ Y
        sigma2 = ((Y - A @ coef) ** 2).sum(0) / dof
        out[0, cols] = coef[1]
        out[1, cols] = np.sqrt(sigma2 * var_rate)
        out[2, cols] = np.hypot(coef[2], coef[3]) 
    return out.reshape(3, ny, nx)
 
 
def deramp(stack, good, max_pts=50_000):
    T, ny, nx = stack.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    xx = (xx - nx / 2) / nx  
    yy = (yy - ny / 2) / ny
    idx = np.flatnonzero(good.ravel())
    if len(idx) > max_pts:  
        idx = np.random.default_rng(0).choice(idx, max_pts, replace=False)
    G = np.column_stack([np.ones(len(idx)), xx.ravel()[idx], yy.ravel()[idx]])
    out = np.empty_like(stack)
    for t in range(T):
        a, b, c = np.linalg.lstsq(G, stack[t].ravel()[idx].astype("float64"), rcond=None)[0]
        out[t] = stack[t] - (a + b * xx + c * yy)
    return out
 
 
def make_cells(stack, good, rate, block, tf):
    T, ny, nx = stack.shape
    xs, ys, n, med, p25, p75, series = [], [], [], [], [], [], []
    need = max(10, int(0.3 * block * block))
    for i in range(0, ny, block):
        for j in range(0, nx, block):
            g = good[i : i + block, j : j + block]
            if g.sum() < need:
                continue
            r = rate[i : i + block, j : j + block][g]
            s = stack[:, i : i + block, j : j + block][:, g]
            h, w = g.shape
            xs.append(tf[0] + (j + w / 2) * tf[1])
            ys.append(tf[3] + (i + h / 2) * tf[5])
            n.append(int(g.sum()))
            med.append(np.median(r))
            p25.append(np.percentile(r, 25))
            p75.append(np.percentile(r, 75))
            series.append(np.median(s, axis=1))
    return dict(x=np.array(xs), y=np.array(ys), n=np.array(n), rate=np.array(med),
                p25=np.array(p25), p75=np.array(p75), series=np.array(series, dtype="float32"))
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("--absolute", action="store_true",
                    help="keep the area-wide shift (relative to the far-away frame reference)")
    ap.add_argument("--coarsen", type=int, default=1, help="average N x N pixels first")
    ap.add_argument("--frame", help="only process this frame, e.g. F08882")
    ap.add_argument("--cell", type=float, default=450,
                    help="size in metres of the neighbourhood blocks saved for the website")
    ap.add_argument("--deramp", action="store_true", help="remove a best-fit tilt per date")
    args = ap.parse_args()
    outdir = Path("data/products") / args.region
    outdir.mkdir(parents=True, exist_ok=True)
    relative = not args.absolute
    tag = "_deramp" if args.deramp else ""
 
    for frame, items in load(args.region, args.coarsen, args.frame or "F").items():
        print(f"{frame}: {len(items)} files, {items[0]['disp'].shape} pixels at {30 * args.coarsen} m")
        dates, stack, masks = chain(items)
        if len(dates) < 6:
            print("  too few chained dates for a rate, skipping")
            continue
        good = (masks[1:].mean(0) >= min_good_fraction) & np.isfinite(stack).all(0)
        print(f"  {len(dates)} dates, {dates[0].date()} to {dates[-1].date()}")
        print(f"  good pixels: {good.mean():.0%} of the crop")
        if not good.any():
            continue
 
        common = np.median(stack[:, good], axis=1)
        cc = np.linalg.lstsq(design(dates), common, rcond=None)[0]
        print(f"  area-wide trend (frame-relative, not trusted): {cc[1]:.1f} mm/yr")
        print(f"  area-wide seasonal amplitude (same caveat): {np.hypot(cc[2], cc[3]):.1f} mm")
        if relative:
            stack = stack - common[:, None, None]
            print("  removed it; rates below are relative to the area median")
        if args.deramp:
            stack = deramp(stack, good)
            print("  removed a best-fit tilt from every date")
 
        rate, se, amp = fit(dates, stack, good)
        p = lambda a, q: float(np.nanpercentile(a, q))
        print(f"  rate mm/yr: median {p(rate, 50):.1f}, 5th {p(rate, 5):.1f}, 95th {p(rate, 95):.1f}")
        print(f"  median formal error: {p(se, 50):.1f} mm/yr")
        print(f"  median seasonal amplitude: {p(amp, 50):.1f} mm")
 
        tf = items[0]["transform"]
        with rasterio.open(
            outdir / f"{frame}{tag}_rate.tif", "w", driver="GTiff",
            height=rate.shape[0], width=rate.shape[1], count=3, dtype="float32",
            crs=CRS.from_wkt(items[0]["wkt"]), transform=Affine.from_gdal(*tf), nodata=np.nan,
        ) as dst:
            dst.write(rate.astype("float32"), 1)
            dst.write(se.astype("float32"), 2)
            dst.write(amp.astype("float32"), 3)
 
        series = stack[:, good]
        ts = dict(
            frame=frame,
            reference=("area median removed per date" if relative else "frame reference point")
            + (", best-fit tilt removed per date" if args.deramp else ""),
            units="mm, line-of-sight, + = toward satellite (roughly up)",
            n_pixels=int(good.sum()),
            dates=[d.date().isoformat() for d in dates],
            p5=np.percentile(series, 5, 1).round(2).tolist(),
            p25=np.percentile(series, 25, 1).round(2).tolist(),
            median=np.median(series, 1).round(2).tolist(),
            p75=np.percentile(series, 75, 1).round(2).tolist(),
            p95=np.percentile(series, 95, 1).round(2).tolist(),
            area_wide_shift=common.round(2).tolist(),
        )
        (outdir / f"{frame}{tag}_timeseries.json").write_text(json.dumps(ts, indent=1))
 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        lim = max(abs(p(rate, 2)), abs(p(rate, 98)), 1)
        ny, nx = rate.shape
        ext = [tf[0], tf[0] + nx * tf[1], tf[3] + ny * tf[5], tf[3]]
        im = ax1.imshow(rate, cmap="RdBu", vmin=-lim, vmax=lim, extent=ext)
        what = ("relative to area median" if relative else "frame-relative") + (", tilt removed" if args.deramp else "")
        fig.colorbar(im, ax=ax1, label=f"LOS rate, mm/yr, {what}\n(blue = toward satellite ≈ up)")
        ax1.set_title(f"{args.region} ({frame}), {dates[0].date()} to {dates[-1].date()}")
        ax1.set_xticks([]); ax1.set_yticks([])
        ax2.fill_between(dates, ts["p5"], ts["p95"], alpha=0.2, label="90% of pixels")
        ax2.fill_between(dates, ts["p25"], ts["p75"], alpha=0.4, label="middle 50% of pixels")
        if relative:
            ax2.plot(dates, common, color="gray", lw=1, ls="--",
                     label="area-wide shift (removed, not trusted)")
        else:
            ax2.plot(dates, ts["median"], "o-", ms=3, label="median")
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.set_ylabel("cumulative LOS displacement (mm)")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"{frame}{tag}_quicklook.png", dpi=120)
        np.savez_compressed(outdir / f"{frame}{tag}_rate.npz", rate=rate, se=se, amp=amp,
                            transform=np.array(tf), crs_wkt=items[0]["wkt"])
        px = abs(float(tf[1]))  
        block = max(1, round(args.cell / px))
        cells = make_cells(stack, good, rate, block, tf)
        np.savez_compressed(outdir / f"{frame}{tag}_cells.npz", **cells, crs_wkt=items[0]["wkt"],
                            cell_m=block * px, dates=np.array([d.date().isoformat() for d in dates]))
        print(f"  {len(cells['n'])} neighbourhood blocks of {block * px:.0f} m saved for the website")
        print("  wrote", outdir)
 
 
if __name__ == "__main__":
    main()
 