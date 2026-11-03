"""8bitdiffuser 64x (SD 1.5 LoRA PX64NOCAP_epoch_10, trigger pixel_art, draws on an 8 px grid) as a last pass over finished
H3 or Wan Fun frames: the video model gives pose and character, SD 1.5 img2img at low denoise redraws each frame on an exact
8x grid, and the /8 downscale (with grid phase search) is then close to lossless.
Source frames: work/<src>_frames/r<row>_c<col>.png when gen_wan_walk.py made them; otherwise the middle frame of each 5-frame hold of a gen_h3_sheet.py run (output/h3_sheet/<src>_r<row>_*.png) or a
gen_wan_fun.py run (not in this repo; output/wan_fun/<src>_r<row>/f_*.png), 384x480 = 48x64 cell + 8 px pad at 6x.
--init hd: the frame resized to 8x (512x640, Lanczos); --init px: first k-centroid /6, then nearest x8 (what we have now, redrawn).
Writes work/<tag>_1x.png, _raw.png, _prompt.json (engine "px64").
--align frame|row (sheet mode): shift the 8x frames to the window that keeps the most outline ink (ink_align.py).
--oe N: PixelOE outline expansion (thickness N) on each 8x frame before the pass (gen_pixeloe.py).
--mode batch: the 4 frames of a row as one batch (different noise per frame); same: one run per frame, same seed;
canvas: the 4 frames side by side in one 2048x640 image, redrawn together; sheet: all 16 frames in one 2048x2560 image.
Usage: python gen_px64_pass.py <tag> --src h3_b6_refhead [--wan] [--denoise 0.45] [--init hd|px] [--mode batch|same|canvas] [--seed 1]
"""
import argparse, json, os
import numpy as np
from PIL import Image
import comfy as g
import gen_h3_sheet as h3
from pixelops import kcentroid_down

CW, CH, PAD, K_SRC, K = 48, 64, 8, 6, 8
W, H = (CW + 2 * PAD) * K, (CH + 2 * PAD) * K          # 512x640
LORA = "PX64NOCAP_epoch_10.safetensors"
import character
POS = ("pixel_art, chibi game character, " + character.DESCRIPTION + ", full body, standing, plain flat light gray background")
NEG = "blurry, photo, realistic, 3d, text, watermark, extra limbs, deformed, scenery"
R = 5


def source_frames(src, row, wan):
    picked = os.path.join(h3.D, src + "_frames")                 # gen_wan_walk.py output: 4 picked frames per row
    if os.path.exists(os.path.join(picked, f"r{row}_c0.png")):
        return [Image.open(os.path.join(picked, f"r{row}_c{c}.png")).convert("RGB") for c in range(4)]
    if wan:
        d = os.path.join(g.COUT, "wan_fun", f"{src}_r{row}"); files = sorted(f for f in os.listdir(d) if f.startswith("f_"))
    else:
        d = os.path.join(g.COUT, "h3_sheet"); files = sorted(f for f in os.listdir(d) if f.startswith(f"{src}_r{row}_"))
    return [Image.open(os.path.join(d, files[k * R + R // 2])).convert("RGB") for k in range(4)]


def init_image(im, mode, oe=0):
    if mode == "px":
        return kcentroid_down(im.resize(((CW + 2 * PAD) * K_SRC, (CH + 2 * PAD) * K_SRC), Image.LANCZOS), K_SRC).resize((W, H), Image.NEAREST)
    im = im.resize((W, H), Image.LANCZOS)
    if oe:
        from gen_pixeloe import outline_expand
        im = outline_expand(im, K, oe)
    return im


def phase_down(im):
    """/8 downscale at the grid phase where the blocks are most uniform (the model's grid need not start at 0,0)."""
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    best = None
    for oy in range(K):
        for ox in range(K):
            h, w = (a.shape[0] - oy) // K, (a.shape[1] - ox) // K
            b = a[oy:oy + h * K, ox:ox + w * K].reshape(h, K, w, K, 3)
            v = b.var(axis=(1, 3)).mean()
            if best is None or v < best[0]:
                best = (v, ox, oy)
    _, ox, oy = best
    small = kcentroid_down(im.crop((ox, oy, im.width, im.height)), K, detail_bias=True)
    full = Image.new("RGB", (W // K, H // K), h3.GREY); full.paste(small, (0, 0))
    return full.crop((PAD, PAD, PAD + CW, PAD + CH)), (ox, oy)


def graph(names, seed, denoise, prefix, lora_strength):
    G = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned.ckpt"}},
         "2": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": LORA,
                                                      "strength_model": lora_strength, "strength_clip": lora_strength}},
         "3": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": ["2", 1], "stop_at_clip_layer": -2}},
         "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": POS}},
         "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": NEG}}}
    prev = None
    for k, n in enumerate(names):
        G[f"i{k}"] = {"class_type": "LoadImage", "inputs": {"image": n}}
        if prev is None:
            prev = [f"i{k}", 0]
        else:
            G[f"b{k}"] = {"class_type": "ImageBatch", "inputs": {"image1": prev, "image2": [f"i{k}", 0]}}; prev = [f"b{k}", 0]
    G["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": prev, "vae": ["1", 2]}}
    G["7"] = {"class_type": "KSampler", "inputs": {"model": ["2", 0], "seed": seed, "steps": 20, "cfg": 6.0, "sampler_name": "dpmpp_2m_sde",
                                                   "scheduler": "karras", "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                                                   "denoise": denoise}}
    G["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    G["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return G


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--src", required=True)
    ap.add_argument("--wan", action="store_true", help="the source is a gen_wan_fun.py run")
    ap.add_argument("--denoise", type=float, default=0.45); ap.add_argument("--init", default="hd", choices=["hd", "px"])
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--rows", default="0,1,2,3")
    ap.add_argument("--mode", default="batch", choices=["batch", "same", "canvas", "sheet"])
    ap.add_argument("--oe", type=int, default=0, help="PixelOE outline expansion thickness before the pass (0 = off)")
    ap.add_argument("--align", choices=["frame", "row"], help="sheet mode: shift each frame (or row) so the most outline ink lands inside SD's 8x8 blocks (ink_align.py)")
    a = ap.parse_args()
    sheet = Image.new("RGB", (CW * 4, CH * 4), h3.GREY)
    raw = Image.new("RGB", (W * 2, H * 2), h3.GREY)
    total, phases = 0, []
    rows = [int(x) for x in a.rows.split(",")]
    if a.mode == "sheet":
        can = Image.new("RGB", (W * 4, H * 4), h3.GREY)
        for r in rows:
            inits = [init_image(f, a.init, a.oe) for f in source_frames(a.src, r, a.wan)]
            if a.align:
                from ink_align import best_offset, shift
                groups = [[im] for im in inits] if a.align == "frame" else [inits]
                picks = [best_offset(gr, K)[:2] for gr in groups]
                picks = picks * 4 if a.align == "row" else picks
                inits = [shift(im, ox, oy) for im, (ox, oy) in zip(inits, picks)]
                print(f"row {r}: 8x offsets {picks}", flush=True)
            for c, im in enumerate(inits):
                can.paste(im, (c * W, r * H))
        t, o = g.wait(g.post(graph([h3.upload(can, f"px64_{a.tag}_sheet")], a.seed, a.denoise, f"px64/{a.tag}_sheet", a.lora_strength)))
        big = Image.open(os.path.join(g.COUT, o["9"]["images"][0].get("subfolder", ""), o["9"]["images"][0]["filename"])).convert("RGB")
        for r in rows:
            for c in range(4):
                cell, ph = phase_down(big.crop((c * W, r * H, (c + 1) * W, (r + 1) * H))); phases.append(ph)
                sheet.paste(cell, (c * CW, r * CH))
        big.resize((W * 2, H * 2), Image.LANCZOS).save(os.path.join(h3.D, a.tag + "_raw.png")); total = t
        rows = []
        print(f"sheet: {t:.0f} s, phases {sorted(set(phases))}", flush=True)
    for r in rows:
        inits = [init_image(f, a.init, a.oe) for f in source_frames(a.src, r, a.wan)]
        load = lambda o: [Image.open(os.path.join(g.COUT, i.get("subfolder", ""), i["filename"])).convert("RGB") for i in o["9"]["images"]]
        if a.mode == "canvas":
            can = Image.new("RGB", (W * 4, H)); [can.paste(im, (c * W, 0)) for c, im in enumerate(inits)]
            t, o = g.wait(g.post(graph([h3.upload(can, f"px64_{a.tag}_r{r}")], a.seed, a.denoise, f"px64/{a.tag}_r{r}", a.lora_strength)))
            big = load(o)[0]; outs = [big.crop((c * W, 0, (c + 1) * W, H)) for c in range(4)]
        elif a.mode == "same":
            outs, t = [], 0
            for c, im in enumerate(inits):
                tc, o = g.wait(g.post(graph([h3.upload(im, f"px64_{a.tag}_r{r}_{c}")], a.seed, a.denoise, f"px64/{a.tag}_r{r}_{c}", a.lora_strength)))
                outs += load(o); t += tc
        else:
            names = [h3.upload(im, f"px64_{a.tag}_r{r}_{c}") for c, im in enumerate(inits)]
            t, o = g.wait(g.post(graph(names, a.seed, a.denoise, f"px64/{a.tag}_r{r}", a.lora_strength))); outs = load(o)
        total += t
        for c, im in enumerate(outs):
            cell, ph = phase_down(im); phases.append(ph)
            sheet.paste(cell, (c * CW, r * CH))
            raw.paste(im.resize((W // 2, H // 2), Image.LANCZOS), (c * W // 2, r * H // 2))
        print(f"row {r}: {t:.0f} s, phases {phases[-4:]}", flush=True)
    sheet.save(os.path.join(h3.D, a.tag + "_1x.png"))
    if a.mode != "sheet":
        raw.save(os.path.join(h3.D, a.tag + "_raw.png"))
    json.dump({"layout": "64", "engine": "px64", "src": a.src, "wan": a.wan, "denoise": a.denoise, "init": a.init,
               "lora_strength": a.lora_strength, "mode": a.mode, "oe": a.oe, "align": a.align, "seed": a.seed, "seconds": round(total)},
              open(os.path.join(h3.D, a.tag + "_prompt.json"), "w"))
    print(a.tag, f"{total:.0f} s")


if __name__ == "__main__":
    main()
