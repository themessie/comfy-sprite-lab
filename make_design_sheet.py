"""Make a design sheet (turnaround) for a new character, and measure one for character.py.

Make: the base body's four standing frames (column 0 of each row) side by side, front | right | back | left, nearest-upscaled on
light grey, are the "mannequin"; Qwen-Image 2.1 (edit, Turbo 8-step LoRA) dresses it as character.DESCRIPTION in one call, so the
same person comes out from four sides at the mannequin's size and proportions. Seeds differ a lot: make a few and pick one.
Measure: finds the four figures on a design sheet (yours or a made one), prints FRONT_CROP (the front view's box) and NECK (the
narrowest row between head and shoulders, inside the crop) for character.py, and writes <name>_measure.png to check them.
Only the front view feeds the pipeline (round one's H3 reference, and its colours for sheet_cleanup.py --design); the side and back
views are for you and for other tools.

Usage: python make_design_sheet.py Men01 [--seeds 1,2,3] [--style simple]   -> work/design_<body>[_simple]_s<seed>.png + measurements
       python make_design_sheet.py --measure path/to/your_sheet.png
"""
import argparse, os, shutil
import numpy as np
from PIL import Image, ImageDraw
import character, settings
from comfy import post, wait, CIN, COUT

GREY = (216, 216, 216)
ORDER = [(0, "front view, facing the viewer"), (2, "right profile, facing right"), (3, "back view, seen from behind: back of the head, no face"),
         (1, "left profile, facing left")]               # base-body rows: 0 down, 1 left, 2 right, 3 up
PROMPT = ("This is a character turnaround sheet of a mannequin, the same figure four times: from left to right the {views}. Dress and draw "
          "this exact mannequin as a real character: {desc}. Keep exactly the mannequin's proportions, pose, size and place in every view: the "
          "same big head, the same short body, arms and legs where the mannequin has them. Cute chibi game character, clean lineart, flat "
          "colours with two or three tones per material, readable face. Plain light grey background, no text.")

SIMPLE = ("This is a character turnaround sheet of a mannequin, the same figure four times: from left to right the {views}. Dress and draw "
          "this exact mannequin as a character for a tiny 48x64-pixel game sprite: {desc}. Keep exactly the mannequin's proportions, pose, "
          "size and place in every view: the same big head, the same short body, arms and legs where the mannequin has them. Design it to "
          "stay readable when shrunk to 48x64 pixels: very thick, bold, dark outlines around the whole figure and between the clothing "
          "parts, large flat areas of colour with at most two tones per material, simple big shapes, big simple eyes, a simple mouth. "
          "No small details: no freckles, no stitching, no buttons, no laces, no fabric texture, no thin lines, no gradients. "
          "Plain light grey background, no text.")


def mannequin_strip(body, scale=6, gap=24):
    """The base body's standing frames (column 0 of every row), cropped to one shared box, side by side on light grey."""
    s = Image.open(os.path.join(settings.BASE_BODIES, body + ".png")).convert("RGBA")
    cw, ch = s.width // 4, s.height // 4
    cells = [s.crop((0, r * ch, cw, (r + 1) * ch)) for r, _ in ORDER]
    boxes = [c.getbbox() for c in cells]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes); x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    w, h = (x1 - x0) * scale, (y1 - y0) * scale
    can = Image.new("RGBA", (4 * w + 5 * gap * scale // 2, h + 2 * gap * scale // 2), GREY + (255,))
    for i, c in enumerate(cells):
        f = c.crop((x0, y0, x1, y1)).resize((w, h), Image.NEAREST)
        can.alpha_composite(f, (gap * scale // 2 + i * (w + gap * scale // 2), gap * scale // 2))
    return can.convert("RGB")


def qwen_graph(image, prompt, seed, prefix):
    return {
        "10": {"class_type": "LoadImage", "inputs": {"image": image}},
        "20": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "21": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "22": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "23": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["20", 0], "lora_name": "Qwen-Image-2.1-Turbo8-8step-lora.safetensors", "strength_model": 1.0}},
        "25": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["21", 0], "vae": ["22", 0], "prompt": prompt, "negative_prompt": "",
                                                                 "resolution": 1024, "images.image_1": ["10", 0]}},
        "26": {"class_type": "BasicGuider", "inputs": {"model": ["23", 0], "conditioning": ["25", 0]}},
        "27": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "28": {"class_type": "BasicScheduler", "inputs": {"model": ["23", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "29": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "30": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["27", 0], "guider": ["26", 0], "sampler": ["29", 0], "sigmas": ["28", 0],
                                                                 "latent_image": ["25", 2]}},
        "31": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["22", 0]}},
        "51": {"class_type": "SaveImage", "inputs": {"images": ["31", 0], "filename_prefix": prefix}},
    }


def measure(path):
    """The four figures' boxes (left to right), the front view's crop with a margin, and the neck line inside it."""
    im = Image.open(path).convert("RGB"); a = np.asarray(im).astype(int)
    bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0)
    fg = np.abs(a - bg).max(-1) > 30
    cols = fg.sum(0) > max(2, fg.shape[0] // 200)
    runs, start = [], None
    for x, on in enumerate(list(cols) + [False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start > im.width // 40:
                runs.append((start, x))
            start = None
    if not runs:
        raise SystemExit("no figures found")
    figs = []
    for x0, x1 in runs[:4]:
        ys = np.nonzero(fg[:, x0:x1].any(1))[0]; figs.append((x0, int(ys[0]), x1, int(ys[-1]) + 1))
    x0, y0, x1, y1 = figs[0]
    m = max(8, (x1 - x0) // 10)
    crop = (max(0, x0 - m), max(0, y0 - m), min(im.width, x1 + m), min(im.height, y1 + m))
    width = fg[y0:y1, x0:x1].sum(1)
    h = y1 - y0; lo, hi = int(h * 0.2), int(h * 0.55)            # the neck sits between the head and the shoulders
    neck_row = lo + int(np.argmin(width[lo:hi]))
    neck = y0 + neck_row - crop[1]
    return figs, crop, neck


def report(path):
    figs, crop, neck = measure(path)
    im = Image.open(path).convert("RGB"); d = ImageDraw.Draw(im)
    for f in figs:
        d.rectangle(f, outline=(0, 160, 255), width=2)
    d.rectangle(crop, outline=(255, 0, 0), width=3); d.line([(crop[0], crop[1] + neck), (crop[2], crop[1] + neck)], fill=(255, 0, 0), width=3)
    out = os.path.join(settings.WORK, os.path.splitext(os.path.basename(path))[0] + "_measure.png"); im.save(out)
    print(f"{os.path.basename(path)}: {len(figs)} figures; for character.py:\n  FRONT_CROP = {crop}\n  NECK = {neck}\n  (check: {out})")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("body", nargs="?"); ap.add_argument("--seeds", default="1")
    ap.add_argument("--measure", help="only measure this design sheet")
    ap.add_argument("--style", default="detailed", choices=["detailed", "simple"],
                    help="simple: thick dark outlines, flat colours, no small details, made to survive the shrink to 48x64")
    a = ap.parse_args()
    if a.measure:
        report(a.measure); return
    if not a.body:
        raise SystemExit("give a base body name (e.g. Men01) or --measure FILE")
    strip = mannequin_strip(a.body); sp = os.path.join(settings.WORK, f"mannequin_{a.body}.png"); strip.save(sp)
    settings.need_comfy(); os.makedirs(os.path.join(CIN, "sprite_forge"), exist_ok=True)
    shutil.copy(sp, os.path.join(CIN, "sprite_forge", os.path.basename(sp)))
    prompt = (SIMPLE if a.style == "simple" else PROMPT).format(views=", ".join(v for _, v in ORDER[:3]) + " and the " + ORDER[3][1], desc=character.DESCRIPTION)
    for seed in [int(s) for s in a.seeds.split(",")]:
        name = f"design_{a.body}{'_simple' if a.style == 'simple' else ''}_s{seed}"
        t, o = wait(post(qwen_graph(f"sprite_forge/{os.path.basename(sp)}", prompt, seed, f"sprite_forge/{name}")))
        i = o["51"]["images"][0]
        out = os.path.join(settings.WORK, name + ".png")
        shutil.copy(os.path.join(COUT, i.get("subfolder", ""), i["filename"]), out)
        print(f"seed {seed}: {t:.0f} s -> {out}")
        report(out)


if __name__ == "__main__":
    main()
