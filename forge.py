"""forge.py: any animation, any directions, any frame count, any sprite size, with the Wan route.

    character (characters/<name>.json) + animation (animations/<name>.json) + sprite spec (cell size, pixel mode)
      design views  -> one Wan 2.2 clip per direction (loop: start = end image; one-shot: back to the start pose)
      -> frame selection (loop: one cycle found by self-similarity, resampled to N frames; one-shot: N evenly spaced frames)
      -> pixelation at 8x per sprite pixel (direct k-centroid /8, or none = smooth downscale) -> background key + one palette
      -> clean-up rules (outline thinning, specks, spikes, head lock if the animation keeps the head still, selective outline)
      -> work/forge/<tag>/: <tag>_sheet.png (rows = directions, columns = frames), <tag>.json (Aseprite-style frames + frameTags,
         durations from the fps), one preview GIF per direction and one with all directions.

Directions: down, down_left, left, up_left, up, up_right, right, down_right. Down/left/right/up come from the design sheet;
the four diagonals from the character's design_diagonals sheet (make_design_sheet.py --diagonals). --mirror-left builds left-facing
rows by mirroring the right-facing ones (half the Wan time; wrong for asymmetric outfits).
Every clip is cached in work/forge/<tag>/clips.json; re-running reuses it (--regen to render again), so changing the frame count
or the pixel mode later costs no GPU time.

Usage: python forge.py walk [--dirs down,left,right,up] [--frames 8] [--cell 48x64] [--margin 8] [--pixel direct|none]
                            [--seed 2] [--colors 32] [--bg magenta] [--ref-head 0.8] [--tag NAME] [--mirror-left] [--regen]
"""
import argparse, glob, hashlib, json, os, shutil
import numpy as np
from PIL import Image
from scipy import ndimage
import character, settings
from comfy import post, wait, CIN, COUT
from gen_wan_walk import graph as wan_graph, BACKGROUNDS, magentaish
from make_design_sheet import measure
from pixelops import kcentroid_down
import sheet_cleanup as rules

HERE = os.path.dirname(os.path.abspath(__file__))
K = 8                                                  # source pixels per sprite pixel (the Wan canvas is cell x 8)
DIRS = ["down", "down_left", "left", "up_left", "up", "up_right", "right", "down_right"]
VIEW = {"down": ("main", 0), "right": ("main", 1), "up": ("main", 2), "left": ("main", 3),
        "down_left": ("diag", 0), "down_right": ("diag", 1), "up_left": ("diag", 2), "up_right": ("diag", 3)}
FACING = {"down": "front view, facing the viewer", "up": "back view, seen from behind, back of the head, no face visible",
          "left": "side view, facing left", "right": "side view, facing right",
          "down_left": "three-quarter front view, turned toward the viewer and to the left",
          "down_right": "three-quarter front view, turned toward the viewer and to the right",
          "up_left": "three-quarter back view, seen from behind and turned to the left, no face visible",
          "up_right": "three-quarter back view, seen from behind and turned to the right, no face visible"}
PROMPT = ("A chibi game character, {desc}, {facing}, {motion}. {loop} Static camera, the character stays in the centre of the frame "
          "and does not move across it or turn around. A {bg} background, flat colours, clean dark outlines.")
LOOP = {True: "A smooth seamless loop that ends exactly where it started.",
        False: "The movement starts and ends in the same standing pose."}


def background(a, key):
    """Pixels that are background: hue test for the magenta key, else near-background pixels connected to the border."""
    if key == "magenta":
        return magentaish(a)
    bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0)
    near = np.abs(a.astype(int) - bg).max(-1) <= 24
    lab, _ = ndimage.label(near); ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])); ids = ids[ids > 0]
    return np.isin(lab, ids)


def cut_view(sheet, i, head):
    """View i of a 4-figure design sheet as RGBA (background = light pixels connected to the crop's border, so white clothes stay);
    head < 1 shrinks everything above the neck (the narrowest row between head and shoulders)."""
    figs, _, _ = measure(sheet)
    im = Image.open(sheet).convert("RGB"); x0, y0, x1, y1 = figs[i]
    a = np.asarray(im.crop((x0, y0, x1, y1)))
    m = ~background(a, "grey")
    fig = Image.fromarray(np.dstack([a, (m * 255).astype(np.uint8)]), "RGBA")
    if head != 1.0:
        width = m.sum(1); h = len(width); lo, hi = int(h * 0.2), int(h * 0.55); neck = lo + int(np.argmin(width[lo:hi]))
        top = fig.crop((0, 0, fig.width, neck)); body = fig.copy(); body.paste((0, 0, 0, 0), (0, 0, fig.width, neck))
        small = top.resize((max(1, int(top.width * head)), max(1, int(neck * head))), Image.LANCZOS)
        out = Image.new("RGBA", fig.size, (0, 0, 0, 0)); out.alpha_composite(body)
        out.alpha_composite(small, ((fig.width - small.width) // 2, neck - small.height)); fig = out.crop(out.getbbox())
    return fig


def start_image(fig, cw, ch, margin, bgc):
    """The figure on the Wan canvas ((cell + 2 x margin) x 8): about 84 % of the cell tall, centred, feet 2 sprite px above the cell
    bottom (the proportions the 48x64 guide used)."""
    W, H = (cw + 2 * margin) * K, (ch + 2 * margin) * K
    s = min(ch * K * 0.84 / fig.height, cw * K * 0.92 / fig.width)
    f = fig.resize((max(1, round(fig.width * s)), max(1, round(fig.height * s))), Image.LANCZOS)
    can = Image.new("RGBA", (W, H), bgc + (255,)); can.alpha_composite(f, ((W - f.width) // 2, (margin + ch - 2) * K - f.height))
    return can.convert("RGB")


def select(files, n, loop, key="magenta"):
    """Frame indices. Loop: the period P at which the figure repeats best (self-similarity of small grey copies with the background
    blanked: Wan's background drifts and would drown the figure), then n frames evenly over one cycle, starting where the loop closes
    best. The best period, not the shortest good one: half a walk cycle is one step, not the walk. If nothing repeats clearly, n
    frames over the whole clip (it is a loop itself). One-shot: n frames evenly over the clip."""
    m = len(files)
    if not loop:
        return [round(i * (m - 1) / n) for i in range(n)]
    arr = []
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB")).copy(); a[background(a, key)] = 0
        arr.append(np.asarray(Image.fromarray(a).convert("L").resize((40, 50), Image.BILINEAR), np.float32))
    sm = np.stack(arr); d = np.abs(sm[:, None] - sm[None]).mean((2, 3))
    base = d[np.triu_indices(m, 1)].mean()
    scores = {p: np.mean([d[t, t + p] for t in range(1, m - p)]) for p in range(max(4, n // 2), int(0.8 * (m - 1)) + 1)}
    if not scores or min(scores.values()) > 0.7 * base:
        return [round(i * (m - 1) / n) for i in range(n)]
    p = min(scores, key=scores.get)
    t0 = min(range(1, m - p), key=lambda t: d[t, t + p])
    return [t0 + round(i * p / n) for i in range(n)]


def pixelate(im, cw, ch, margin, mode):
    if mode == "none":
        small = im.resize((cw + 2 * margin, ch + 2 * margin), Image.LANCZOS)
    else:
        small = kcentroid_down(im, K, true_colour=True)
    return small.crop((margin, margin, margin + cw, margin + ch))


def finish(sheet_rgb, cw, ch, colors, key, headlock):
    """Background key, one shared palette, clean-up rules; RGBA sheet."""
    a = np.asarray(sheet_rgb).astype(np.uint8); H, W = a.shape[:2]
    clear = np.zeros((H, W), bool)
    for y in range(0, H, ch):
        for x in range(0, W, cw):
            clear[y:y + ch, x:x + cw] = background(a[y:y + ch, x:x + cw], key)
    rgba = np.dstack([a, np.where(clear, 0, 255)]).astype(np.uint8)
    op = rgba[..., 3] > 0
    if colors and op.any():
        pal = Image.fromarray(rgba[op][None, :, :3]).quantize(colors, method=Image.Quantize.MEDIANCUT)
        rgba[op, :3] = np.asarray(pal.convert("RGB"))[0]
    for r, c, cell in rules.cells(rgba, cw, ch):
        cell[:] = rules.silhouette(rules.orphans(rules.thin(cell), 14))
    if headlock:
        for r in range(H // ch):
            row = [rgba[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] for c in range(W // cw)]
            if len(row) > 1:
                rules.headlock(row, max(4, int(ch * 0.33)), targets=range(1, len(row)))
    pal = np.unique(rgba[..., :3][rgba[..., 3] > 0], axis=0)
    for r, c, cell in rules.cells(rgba, cw, ch):
        cell[:] = rules.selout(cell, pal)
    return Image.fromarray(rgba, "RGBA")


def export(sheet, tag, out, dirs, n, cw, ch, fps):
    sheet.save(os.path.join(out, tag + "_sheet.png"))
    frames, tags = {}, []
    for r, dname in enumerate(dirs):
        for c in range(n):
            frames[f"{tag}_{dname}_{c}"] = {"frame": {"x": c * cw, "y": r * ch, "w": cw, "h": ch}, "duration": round(1000 / fps)}
        tags.append({"name": dname, "from": r * n, "to": r * n + n - 1, "direction": "forward"})
    json.dump({"frames": frames, "meta": {"app": "comfy-sprite-lab", "image": tag + "_sheet.png", "size": {"w": sheet.width, "h": sheet.height},
                                          "frameTags": tags}}, open(os.path.join(out, tag + ".json"), "w"), indent=1)
    BG = (52, 58, 66); S = max(2, 256 // ch)
    rows = []
    for r, dname in enumerate(dirs):
        fr = []
        for c in range(n):
            b = Image.new("RGBA", (cw, ch), BG + (255,)); b.alpha_composite(sheet.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)))
            fr.append(b.convert("RGB").resize((cw * S, ch * S), Image.NEAREST))
        fr[0].save(os.path.join(out, f"{tag}_{dname}.gif"), save_all=True, append_images=fr[1:], duration=round(1000 / fps), loop=0)
        rows.append(fr)
    allf = []
    for c in range(n):
        f = Image.new("RGB", (cw * S * len(dirs), ch * S), BG)
        for r in range(len(dirs)):
            f.paste(rows[r][c], (r * cw * S, 0))
        allf.append(f)
    allf[0].save(os.path.join(out, f"{tag}_all.gif"), save_all=True, append_images=allf[1:], duration=round(1000 / fps), loop=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("anim")
    ap.add_argument("--dirs", default="down,left,right,up"); ap.add_argument("--frames", type=int)
    ap.add_argument("--cell", default="48x64"); ap.add_argument("--margin", type=int, default=8)
    ap.add_argument("--pixel", default="direct", choices=["direct", "none"]); ap.add_argument("--colors", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2); ap.add_argument("--bg", default="magenta", choices=sorted(BACKGROUNDS))
    ap.add_argument("--ref-head", type=float, default=0.8); ap.add_argument("--tag")
    ap.add_argument("--mirror-left", action="store_true"); ap.add_argument("--regen", action="store_true")
    a = ap.parse_args()
    anim = json.load(open(os.path.join(HERE, "animations", a.anim + ".json"), encoding="utf-8"))
    n = a.frames or anim["frames"]; cw, ch = (int(v) for v in a.cell.lower().split("x"))
    if ((cw + 2 * a.margin) * K) % 16 or ((ch + 2 * a.margin) * K) % 16:
        raise SystemExit("cell + 2 x margin must be even in both directions (Wan needs sizes divisible by 16)")
    dirs = [d.strip() for d in a.dirs.split(",")]
    bad = [d for d in dirs if d not in DIRS]
    if bad:
        raise SystemExit(f"unknown directions {bad}; use {DIRS}")
    if any(VIEW[d][0] == "diag" for d in dirs) and not character.DESIGN_DIAGONALS:
        raise SystemExit("diagonal directions need design_diagonals in the character file (make_design_sheet.py --diagonals)")
    tag = a.tag or f"{character.NAME}_{a.anim}_{cw}x{ch}"
    out = os.path.join(settings.WORK, "forge", tag); os.makedirs(out, exist_ok=True)
    cache_p = os.path.join(out, "clips.json"); cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    bgc, bg_words = BACKGROUNDS[a.bg]
    W, H = (cw + 2 * a.margin) * K, (ch + 2 * a.margin) * K
    settings.need_comfy(); os.makedirs(os.path.join(CIN, "sprite_forge"), exist_ok=True)
    picked = {}
    for dname in dirs:
        if a.mirror_left and dname in ("left", "up_left", "down_left") and dname.replace("left", "right") in dirs:
            continue
        motion = anim.get("motion_by_dir", {}).get(dname, anim["motion"])
        prompt = PROMPT.format(desc=character.DESCRIPTION, facing=FACING[dname], motion=motion, loop=LOOP[anim["loop"]], bg=bg_words)
        # the key includes the prompt: editing an animation's wording renders that direction again instead of reusing a stale clip
        key = f"{dname}|{a.seed}|{a.bg}|{cw}x{ch}|{a.margin}|{a.ref_head}|{hashlib.sha1(prompt.encode()).hexdigest()[:10]}"
        files = cache.get(key)
        if a.regen or not files or not all(os.path.exists(f) for f in files):
            sheet_path, idx = (character.DESIGN_SHEET, VIEW[dname][1]) if VIEW[dname][0] == "main" else (character.DESIGN_DIAGONALS, VIEW[dname][1])
            start = start_image(cut_view(sheet_path, idx, a.ref_head), cw, ch, a.margin, bgc)
            sname = f"sprite_forge/{tag}_{dname}_start.png"; start.save(os.path.join(CIN, sname)); start.save(os.path.join(out, f"start_{dname}.png"))
            t, o = wait(post(wan_graph(sname, prompt, a.seed, f"sprite_forge/{tag}_{dname}/f", anim.get("clip_frames", 33), width=W, height=H)))
            files = sorted(os.path.join(COUT, i.get("subfolder", ""), i["filename"]) for i in o["13"]["images"])
            cache[key] = files; json.dump(cache, open(cache_p, "w"), indent=1)
            print(f"{dname}: {t:.0f} s, {len(files)} frames", flush=True)
        picked[dname] = select(files, n, anim["loop"])
        cache_frames = [Image.open(files[i]).convert("RGB") for i in picked[dname]]
        picked[dname] = (picked[dname], cache_frames)
    sheet = Image.new("RGB", (cw * n, ch * len(dirs)), bgc)
    for r, dname in enumerate(dirs):
        src = dname
        mirrored = dname not in picked
        if mirrored:
            src = dname.replace("left", "right")
        for c, fr in enumerate(picked[src][1]):
            cell = pixelate(fr, cw, ch, a.margin, a.pixel)
            sheet.paste(cell.transpose(Image.FLIP_LEFT_RIGHT) if mirrored else cell, (c * cw, r * ch))
    final = finish(sheet, cw, ch, a.colors, a.bg if a.bg == "magenta" else "grey", anim.get("headlock", False))
    export(final, tag, out, dirs, n, cw, ch, anim.get("fps", 8))
    json.dump({"anim": a.anim, "dirs": dirs, "frames": n, "cell": [cw, ch], "margin": a.margin, "pixel": a.pixel, "seed": a.seed,
               "bg": a.bg, "ref_head": a.ref_head, "mirror_left": a.mirror_left,
               "picks": {d: v[0] for d, v in picked.items()}}, open(os.path.join(out, tag + "_settings.json"), "w"), indent=1)
    print(tag, "->", out)


if __name__ == "__main__":
    main()
