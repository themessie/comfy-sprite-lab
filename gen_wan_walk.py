"""Wan-only round one: a walk loop per direction with Wan 2.2 I2V (A14B GGUF + lightx2v 4-step LoRAs, Kijai's WanVideoWrapper),
instead of the MiniMax H3 character swap. Apache-2.0 models only, so its results can be shared anywhere.

Each direction's view is cut from the design sheet (front, right, back, left, as make_design_sheet.py measures them), its head
shrunk like round one's --ref-head, and placed at the guide's size and foot line on a 512x640 canvas (48x64 cell + 8 px margin at
8x, the pixel pass's grid). Wan walks in place from that image and back to it (start = end image, a loop). From the 33 frames the
gait is read from the leg spread (width of the lowest fifth of the figure, plus the height difference of the two feet): the widest
stride, the opposite stride half a gait period later, and the most closed pose between them become columns 1, 3 and 0 (column 2 =
column 0). Writes work/<tag>_frames/r<row>_c<col>.png for gen_px64_pass.py --frames, and work/<tag>_wan_r<row>.png contact strips.

Also writes work/<tag>_1x.png (each picked frame /8 directly: k-centroid with true colours, no detail bias) for sheet_post.py.
With a simple design (make_design_sheet.py --style simple) this direct /8 beat every pixel-pass variant in our tests, so the
pass is optional on this route. Round two (--start-sheet) starts Wan from finished sprites at 8x instead. Rows in --high-steps (default front and back, whose walks
barely move the legs) get a high-knee prompt.
Usage: python gen_wan_walk.py owens_wan [--design sheet.png] [--rig rig_Men01_body_big] [--seed 1] [--rows 0,1,2,3] [--ref-head 0.8]
Then:  python gen_px64_pass.py owens_wan_px --frames owens_wan --denoise 0.45 --mode sheet --oe 3 --align row   (and steps 6-7)
"""
import argparse, json, os, shutil
import numpy as np
from PIL import Image
from scipy import ndimage
import character, settings
from comfy import post, wait, CIN, COUT
from make_design_sheet import measure
from pixelops import kcentroid_down

D = settings.WORK
CW, CH, PAD, K = 48, 64, 8, 8
W, H = (CW + 2 * PAD) * K, (CH + 2 * PAD) * K          # 512x640
GREY = (200, 200, 200)
# Background of the Wan canvas. Magenta by default: light grey looked like white clothing (a cream blouse came out 4-5 levels from
# the grey and was keyed away with it); magenta is rare on characters. Use grey or green for magenta/pink characters.
BACKGROUNDS = {"magenta": ((255, 0, 255), "plain flat solid magenta"), "green": ((0, 190, 60), "plain flat solid bright green"),
               "grey": (GREY, "plain flat light gray")}
BGC, BG_WORDS = BACKGROUNDS["magenta"]
VIEW_OF_ROW = {0: 0, 1: 3, 2: 1, 3: 2}                  # sheet rows down/left/right/up -> design views front/right/back/left
WALK = ["walking in place toward the viewer, front view, facing the camera, at a calm pace with small natural steps, arms hanging close "
        "to the body and swinging a little, the body stays upright, no jumping, no hopping",
        "walking in place to the left, side view, facing left",
        "walking in place to the right, side view, facing right",
        "walking in place away from the viewer, back view, seen from behind, back of the head, no face visible, at a calm pace with small "
        "natural steps, arms hanging close to the body and swinging a little, the body stays upright, no jumping, no hopping"]
PROMPT = ("A chibi game character, {desc}, {walk}. A smooth looping walk cycle with clear, alternating steps: one leg forward while "
          "the other pushes off behind, feet lifting off the ground, arms swinging opposite to the legs. Static camera, the character "
          "stays in the centre of the frame and does not move across it or turn around. A {bg} background, flat colours, "
          "clean dark outlines.")
HIGH = (" Marching walk with high knee lifts: each foot clearly leaves the ground and the knee comes up high, the legs never stay "
        "side by side for long.")
NEG = ("camera movement, zoom, panning, character moving across the frame, turning around, facing the camera from behind, blurry, "
       "3d render, photo, extra limbs, deformed legs, text, watermark, background scenery")


def magentaish(a):
    """Key colour test for the magenta background: red and blue both well above green. Wan lets the background drift from pink to
    purple across the frame, so a distance to one colour does not work; dark reds (hair) have low blue and stay."""
    a = a.astype(int)
    return (a[..., 0] - a[..., 1] > 70) & (a[..., 2] - a[..., 1] > 70)


def fg_mask(a, tol=30):
    bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0)
    if bg[0] > 150 and bg[2] > 150 and bg[1] < 100:          # magenta key
        return ~magentaish(a)
    return np.abs(a.astype(int) - bg).max(-1) > tol


def view_image(design, i, head, target_h, foot_y):
    """Design view i (0 front, 1 right, 2 back, 3 left) on the 512x640 canvas: head above the neck shrunk to `head`, scaled to
    target_h px tall, centred, feet on foot_y."""
    figs, _, _ = measure(design)
    im = Image.open(design).convert("RGB"); x0, y0, x1, y1 = figs[i]
    fig = im.crop((x0, y0, x1, y1)); a = np.asarray(fig)
    # background = near-background pixels connected to the crop's border; enclosed light areas (a white blouse) stay part of the figure
    near = ~fg_mask(np.asarray(im))[y0:y1, x0:x1]
    lab_, _ = ndimage.label(near); ids = np.unique(np.concatenate([lab_[0], lab_[-1], lab_[:, 0], lab_[:, -1]])); ids = ids[ids > 0]
    m = ~np.isin(lab_, ids)
    rgba = np.dstack([a, (m * 255).astype(np.uint8)]); fig = Image.fromarray(rgba, "RGBA")
    if head != 1.0:
        width = m.sum(1); h = len(width); lo, hi = int(h * 0.2), int(h * 0.55)
        neck = lo + int(np.argmin(width[lo:hi]))
        top = fig.crop((0, 0, fig.width, neck)); body = fig.copy(); body.paste((0, 0, 0, 0), (0, 0, fig.width, neck))
        small = top.resize((max(1, int(top.width * head)), max(1, int(neck * head))), Image.LANCZOS)
        out = Image.new("RGBA", fig.size, (0, 0, 0, 0)); out.alpha_composite(body)
        out.alpha_composite(small, ((fig.width - small.width) // 2, neck - small.height)); fig = out.crop(out.getbbox())
    s = target_h / fig.height
    fig = fig.resize((max(1, round(fig.width * s)), target_h), Image.LANCZOS)
    can = Image.new("RGBA", (W, H), BGC + (255,)); can.alpha_composite(fig, ((W - fig.width) // 2, foot_y - fig.height))
    return can.convert("RGB")


def rig_geometry(rig):
    """Figure height and foot line (in 8x canvas px) of the guide's standing frame, so Wan's character matches round one's size."""
    p = os.path.join(D, rig + "_1x.png")
    if not os.path.exists(p):
        return int(56 * K), (PAD + CH - 2) * K
    a = np.asarray(Image.open(p).convert("RGB"))[0:CH, 0:CW]; m = fg_mask(a)
    ys = np.nonzero(m.any(1))[0]
    return int((ys[-1] - ys[0] + 1) * K), int((PAD + ys[-1] + 1) * K)


def graph(img, prompt, seed, prefix, frames, pixel_lora=0.0, width=None, height=None, neg=None, steps=4, shift=5.0):
    lora_hi = "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
    lora_lo = "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
    G = {
        "1": {"class_type": "LoadImage", "inputs": {"image": img}},
        "2": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "wan_2.1_vae.safetensors", "precision": "bf16"}},
        "3": {"class_type": "WanVideoTextEncodeCached", "inputs": {"model_name": "umt5-xxl-enc-fp8_e4m3fn.safetensors", "precision": "bf16",
              "positive_prompt": prompt, "negative_prompt": neg or NEG, "quantization": "disabled", "use_disk_cache": True, "device": "gpu"}},
        "4": {"class_type": "WanVideoBlockSwap", "inputs": {"blocks_to_swap": 15, "offload_img_emb": False, "offload_txt_emb": False}},
        "5": {"class_type": "WanVideoLoraSelect", "inputs": {"lora": lora_hi, "strength": 1.0, "merge_loras": False}},
        "6": {"class_type": "WanVideoLoraSelect", "inputs": {"lora": lora_lo, "strength": 1.0, "merge_loras": False}},
        "7": {"class_type": "WanVideoModelLoader", "inputs": {"model": "Wan2.2-I2V-A14B-HighNoise-Q5_K_S.gguf", "base_precision": "fp16", "quantization": "disabled",
              "load_device": "offload_device", "attention_mode": "sdpa", "block_swap_args": ["4", 0], "lora": ["5", 0]}},
        "8": {"class_type": "WanVideoModelLoader", "inputs": {"model": "Wan2.2-I2V-A14B-LowNoise-Q5_K_S.gguf", "base_precision": "fp16", "quantization": "disabled",
              "load_device": "offload_device", "attention_mode": "sdpa", "block_swap_args": ["4", 0], "lora": ["6", 0]}},
        "9": {"class_type": "WanVideoImageToVideoEncode", "inputs": {"width": width or W, "height": height or H, "num_frames": frames, "noise_aug_strength": 0.0,
              "start_latent_strength": 1.0, "end_latent_strength": 1.0, "force_offload": True, "vae": ["2", 0], "start_image": ["1", 0],
              "end_image": ["1", 0]}},
        "10": {"class_type": "WanVideoSampler", "inputs": {"model": ["7", 0], "image_embeds": ["9", 0], "text_embeds": ["3", 0], "steps": steps, "cfg": 1.0,
               "shift": shift, "seed": seed, "force_offload": True, "scheduler": "euler", "riflex_freq_index": 0, "start_step": 0, "end_step": steps // 2}},
        "11": {"class_type": "WanVideoSampler", "inputs": {"model": ["8", 0], "image_embeds": ["9", 0], "text_embeds": ["3", 0], "samples": ["10", 0], "steps": steps,
               "cfg": 1.0, "shift": shift, "seed": seed, "force_offload": True, "scheduler": "euler", "riflex_freq_index": 0, "start_step": steps // 2, "end_step": steps}},
        "12": {"class_type": "WanVideoDecode", "inputs": {"vae": ["2", 0], "samples": ["11", 0], "enable_vae_tiling": False, "tile_x": 272, "tile_y": 272,
               "tile_stride_x": 144, "tile_stride_y": 128}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }
    if pixel_lora:   # Wan 2.1 2D Pixel Toolkit (Civitai 1656375, Apache-2.0) on the low-noise model, with its trigger words
        G["15"] = {"class_type": "WanVideoLoraSelect", "inputs": {"lora": r"Wan2.1\pixel_toolkit\wan2.1_pixel_5.safetensors", "strength": pixel_lora,
                                                                 "prev_lora": ["6", 0], "merge_loras": False}}
        G["8"]["inputs"]["lora"] = ["15", 0]
        G["3"]["inputs"]["positive_prompt"] = "SYK006, pixel_art, " + prompt
    return G


def sprite_start(sheet, r):
    """Round two: the row's standing sprite (column 0) of a finished 48x64 sheet, nearest-upscaled 8x onto the 512x640 canvas at the
    same place (8 px margin), so each sprite pixel fills one 8x8 cell of Wan's latent grid."""
    spr = Image.open(sheet).convert("RGBA").crop((0, r * CH, CW, (r + 1) * CH))
    can = Image.new("RGBA", (CW + 2 * PAD, CH + 2 * PAD), BGC + (255,)); can.alpha_composite(spr, (PAD, PAD))
    return can.convert("RGB").resize((W, H), Image.NEAREST)


def foot_lift(im):
    """How far one foot is lifted above the other (front/back walks step toward the camera: one foot rises, the legs do not
    splay), minus a penalty for sideways splay and flung-out arms, relative to the figure height."""
    a = np.asarray(im.convert("RGB")); m = fg_mask(a)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return 0.0
    y0, y1 = ys.min(), ys.max(); h = y1 - y0 + 1
    low = m[y1 - h // 5:y1 + 1]; cols = np.nonzero(low.any(0))[0]
    if len(cols) < 2:
        return 0.0
    mid = (cols[0] + cols[-1]) // 2
    feet = [np.nonzero(m[:, s].any(1))[0] for s in (slice(0, mid), slice(mid, None))]
    lift = abs(int(feet[0].max()) - int(feet[1].max())) if all(len(f) for f in feet) else 0
    splay = (cols[-1] - cols[0]) / h
    arms = (np.ptp(np.nonzero(m[y0 + h * 2 // 5:y0 + h * 3 // 5].any(0))[0]) if m[y0 + h * 2 // 5:y0 + h * 3 // 5].any() else 0) / h
    return float(lift / h - 0.5 * max(0.0, splay - 0.35) - 0.5 * max(0.0, arms - 0.55))


def leg_spread(im):
    a = np.asarray(im.convert("RGB")); m = fg_mask(a)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return 0.0
    y0, y1 = ys.min(), ys.max(); low = m[y1 - (y1 - y0) // 5:y1 + 1]
    cols = np.nonzero(low.any(0))[0]; width = cols[-1] - cols[0] if len(cols) else 0
    mid = (cols[0] + cols[-1]) // 2 if len(cols) else 0
    feet = [np.nonzero(m[:, s].any(1))[0] for s in (slice(0, mid), slice(mid, None))]
    lift = abs(int(feet[0].max()) - int(feet[1].max())) if all(len(f) for f in feet) else 0
    return float(width + 2 * lift)


def pick(files, metric="spread"):
    """(neutral, stride A, stride B) frame indices from the leg-spread signal (or foot lift, for front/back walks)."""
    fn = foot_lift if metric == "lift" else leg_spread
    s = np.array([fn(Image.open(f)) for f in files]); n = len(s)
    inner = range(2, n - 2)
    x = s - s.mean(); ac = [float((x[:-lag] * x[lag:]).mean()) for lag in range(3, n // 2)]
    half = 3 + int(np.argmax(ac)) if ac else 6                 # legs-apart peaks repeat every half gait period (left, right, ...)
    a = max(inner, key=lambda i: s[i])
    cands = [i for i in inner if abs(abs(i - a) - half) <= 2]
    b = max(cands, key=lambda i: s[i]) if cands else min(n - 3, a + half)
    lo, hi = sorted((a, b)); between = range(lo + 1, hi) if hi - lo > 1 else inner
    neutral = min(between, key=lambda i: s[i])
    return neutral, a, b, s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--design", default=character.DESIGN_SHEET)
    ap.add_argument("--rig", default="rig_Men01_body_big"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rows", default="0,1,2,3"); ap.add_argument("--ref-head", type=float, default=0.8)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=4, help="sampler steps, split evenly between the high- and low-noise model")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--walk0", help="replace the front-row walk phrase"); ap.add_argument("--walk3", help="replace the back-row walk phrase")
    ap.add_argument("--repick", action="store_true", help="no generation: pick frames again from this tag's existing Wan clips")
    ap.add_argument("--bg", default="magenta", choices=sorted(BACKGROUNDS), help="Wan canvas background (keyed out afterwards)")
    ap.add_argument("--pick", default="auto", choices=["auto", "spread", "lift"],
                    help="frame picking: leg spread (sides) or foot lift (front/back); auto = lift for rows 0 and 3")
    ap.add_argument("--pixel-lora", type=float, default=0.0, help="Wan 2.1 2D Pixel Toolkit LoRA strength on the low-noise model (0 = off)")
    ap.add_argument("--pixel-start", action="store_true",
                    help="pixelate the design view first (k-centroid /8 to the sprite grid, then nearest x8), so Wan starts from pixel art")
    ap.add_argument("--start-sheet", help="round two: start from this finished sheet's standing sprites (x8) instead of the design")
    ap.add_argument("--high-steps", default="", help="rows that get the high-knee prompt (front/back walks move the legs little); '' for none")
    a = ap.parse_args()
    global BGC, BG_WORDS
    BGC, BG_WORDS = BACKGROUNDS[a.bg]
    settings.need_comfy()
    target_h, foot_y = rig_geometry(a.rig)
    fd = os.path.join(D, a.tag + "_frames"); os.makedirs(fd, exist_ok=True); os.makedirs(os.path.join(CIN, "sprite_forge"), exist_ok=True)
    picks = {}
    sheet = Image.new("RGB", (CW * 4, CH * 4), BGC)
    if a.rows != "0,1,2,3" and os.path.exists(os.path.join(D, a.tag + "_1x.png")):
        sheet.paste(Image.open(os.path.join(D, a.tag + "_1x.png")).convert("RGB"))
    for r in [int(x) for x in a.rows.split(",")]:
        start = sprite_start(a.start_sheet, r) if a.start_sheet else view_image(a.design, VIEW_OF_ROW[r], a.ref_head, target_h, foot_y)
        if a.pixel_start and not a.start_sheet:
            start = kcentroid_down(start, K, detail_bias=True).resize((W, H), Image.NEAREST)
        sname = f"sprite_forge/{a.tag}_start_r{r}.png"; start.save(os.path.join(CIN, sname)); start.save(os.path.join(fd, f"start_r{r}.png"))
        walk = {0: a.walk0, 3: a.walk3}.get(r) or WALK[r]
        prompt = PROMPT.format(desc=character.DESCRIPTION, walk=walk, bg=BG_WORDS) + (HIGH if str(r) in a.high_steps.split(",") else "")
        if a.repick:
            import glob
            t, files = 0.0, sorted(glob.glob(os.path.join(COUT, "sprite_forge", f"{a.tag}_r{r}", "f_*.png")))
        else:
            t, o = wait(post(graph(sname, prompt, a.seed, f"sprite_forge/{a.tag}_r{r}/f", a.frames, a.pixel_lora, steps=a.steps, shift=a.shift)))
            files = sorted(os.path.join(COUT, i.get("subfolder", ""), i["filename"]) for i in o["13"]["images"])
        metric = a.pick if a.pick != "auto" else ("lift" if r in (0, 3) else "spread")
        neutral, sa, sb, s = pick(files, metric)
        for c, idx in enumerate((neutral, sa, neutral, sb)):
            Image.open(files[idx]).convert("RGB").save(os.path.join(fd, f"r{r}_c{c}.png"))
        strip = [Image.open(f).convert("RGB").resize((W // 4, H // 4)) for f in files[::2]]
        cs = Image.new("RGB", (len(strip) * (W // 4), H // 4)); [cs.paste(im, (i * (W // 4), 0)) for i, im in enumerate(strip)]
        cs.save(os.path.join(D, f"{a.tag}_wan_r{r}.png"))
        for c, idx in enumerate((neutral, sa, neutral, sb)):                 # direct /8 (round two: no pixel pass needed)
            small = kcentroid_down(Image.open(files[idx]).convert("RGB"), K, detail_bias=False, true_colour=True).crop((PAD, PAD, PAD + CW, PAD + CH))
            sheet.paste(small, (c * CW, r * CH))
        picks[r] = {"neutral": neutral, "stride_a": sa, "stride_b": sb, "spread": [round(float(v), 1) for v in s]}
        print(f"row {r}: {t:.0f} s, {len(files)} frames, neutral {neutral}, strides {sa} / {sb}", flush=True)
    json.dump({"engine": "wan_walk", "design": a.design, "start_sheet": a.start_sheet, "pixel_start": a.pixel_start, "pixel_lora": a.pixel_lora, "steps": a.steps, "shift": a.shift, "pick": a.pick, "bg": a.bg, "walk0": a.walk0, "high_steps": a.high_steps, "seed": a.seed,
               "ref_head": a.ref_head, "picks": picks}, open(os.path.join(D, a.tag + "_wan.json"), "w"), indent=1)
    sheet.save(os.path.join(D, a.tag + "_1x.png"))                            # for sheet_post.py
    json.dump({"layout": "64", "engine": "wan_walk", "seed": a.seed}, open(os.path.join(D, a.tag + "_prompt.json"), "w"))


if __name__ == "__main__":
    main()
