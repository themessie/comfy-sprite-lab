"""MiniMax H3 character swap onto an existing chibi walk (ComfyUI API graph).
<Video 1> = one row of a 32x48 sprite rig (default: the RMXP Men01 base body), each cell padded to 40x56 (figure ~85%, feet inside),
nearest-upscaled 12x (480x672) onto light grey, each frame held R=5 video frames; <Picture 1> = Owens' front view.
The middle frame of each hold is kept, k-centroid /12 back to 40x56 and cropped to the 32x48 cell; the 4 rows make a 1x sheet that
sheet_post.py can slice (writes work/<tag>_1x.png + _prompt.json).
--still R C: test one frame (rig row R, column C repeated for the whole video).
--rows 3 --seed 2 on an existing tag re-rolls only that row and keeps the others (H3 sometimes turns the back view to face the camera
in the later frames; check the back row and re-roll it).
Usage: python gen_h3_sheet.py <tag> [--rig rig_Men01_body | --big --rig rig_Men01_body_big] [--rows 0,1,2,3] [--still 0 1] [--seed 1] [--k 8 --pad 4] [--calm] [--heads 3] [--ref-head 0.8]
"""
import argparse, glob, json, os, shutil, time
from PIL import Image
from comfy import post, wait, CIN, COUT
from pixelops import kcentroid_down
import character, settings
OWENS = character.DESCRIPTION

HERE = os.path.dirname(os.path.abspath(__file__))
D = settings.WORK
CW, CH, PAD, K = 32, 48, 4, 12                 # cell, padding per side, upscale: (32+8) x (48+8) x 12 = 480 x 672
W, H = (CW + 2 * PAD) * K, (CH + 2 * PAD) * K
R, LENGTH = 5, 22                             # 4 holds of 5 frames + 2 spare (H3 lengths are 5 + 17k)
GREY = (200, 200, 200)
CHIBI = ("chibi proportions, big head, about 2.5 heads tall; flat colours, clean lineart; background stays plain flat solid light gray, "
         "empty, no scenery; full body head to toe, feet visible.")
VIEWS = ["walking toward the camera, front view.", "walking to the left, side view, facing left.",
         "walking to the right, side view, facing right.", "walking away from the camera, seen from behind, back view, no face visible."]
CALM = ("The coat reaches the knees. Eyes open, mouth closed, calm neutral expression in every frame, no talking, no blinking. ")
PROMPT = open(os.path.join(HERE, "prompts", "h3_swap.txt"), encoding="utf-8").read()   # "[add description here]" is filled in


def graph(ref, desc, seed, prefix, steps=8, lora=None, lora_strength=1.0):
    g = _graph(ref, desc, seed, prefix, steps)
    if lora:                                   # extra LoRA (e.g. the TFM pixel LoRA) after the character-swap LoRA
        g["39"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["36", 0], "lora_name": lora, "strength_model": lora_strength}}
        g["72"]["inputs"]["model"] = g["74"]["inputs"]["model"] = ["39", 0]
    return g


def _graph(ref, desc, seed, prefix, steps=8):
    return {
        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "41": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["5", 0], "shift_video": 12, "shift_audio": 3}},
        "42": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["41", 0], "attention": "comfy kitchen attention"}},
        "38": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["42", 0], "lora_name": r"h3\minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength_model": 1.0}},
        "36": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["38", 0], "lora_name": r"h3\minimax_h3_character_swap_v1.1_ref2va_fro.safetensors", "strength_model": 1.0}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "30": {"class_type": "LoadImage", "inputs": {"image": ref}},
        "47": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["4", 0], "vae": ["1", 0], "audio_vae": ["2", 0],
                                                                     "ref_images.ref_image_0": ["30", 0],
                                                                     "prompt": PROMPT.replace("[add description here]", desc),
                                                                     "width": W, "height": H, "length": LENGTH, "ref_image_size": "match"}},
        "60": {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": ["47", 0], "latent": ["47", 1], "vae": ["1", 0], "image": None, "frame_idx": 0}},
        "72": {"class_type": "CFGGuider", "inputs": {"model": ["36", 0], "positive": ["60", 0], "negative": ["60", 0], "cfg": 1.0}},
        "73": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "74": {"class_type": "BasicScheduler", "inputs": {"model": ["36", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "33": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "83": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["33", 0], "guider": ["72", 0], "sampler": ["73", 0], "sigmas": ["74", 0], "latent_image": ["47", 1]}},
        "76": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["83", 0]}},
        "81": {"class_type": "VAEDecode", "inputs": {"samples": ["76", 0], "vae": ["1", 0]}},
        "51": {"class_type": "SaveImage", "inputs": {"images": ["81", 0], "filename_prefix": prefix}},
    }


def guide_frame(rig, r, c):
    """One rig cell, padded onto light grey and nearest-upscaled K times."""
    cell = rig.crop((c * CW, r * CH, (c + 1) * CW, (r + 1) * CH))
    bg = cell.getpixel((0, 0))
    px = cell.load()
    for y in range(CH):
        for x in range(CW):
            if px[x, y] == bg:
                px[x, y] = GREY
    pad = Image.new("RGB", (CW + 2 * PAD, CH + 2 * PAD), GREY); pad.paste(cell, (PAD, PAD))
    return pad.resize((W, H), Image.NEAREST)


def reference(head=1.0, neck=character.NECK):
    """Owens' front view (first view of the 4-view sheet) on light grey. head < 1 shrinks everything above the neck line (y in the
    crop), bottom-centred on the neck, so the reference itself has a smaller head."""
    im = Image.open(character.DESIGN_SHEET).convert("RGBA").crop(character.FRONT_CROP)
    if head != 1.0:
        top = im.crop((0, 0, im.width, neck)); body = im.copy(); body.paste((0, 0, 0, 0), (0, 0, im.width, neck))
        small = top.resize((int(top.width * head), int(neck * head)), Image.LANCZOS)
        body.alpha_composite(small, ((im.width - small.width) // 2, neck - small.height)); im = body
    b = Image.new("RGBA", im.size, GREY + (255,)); b.alpha_composite(im)
    return b.convert("RGB")


def upload(im, name):
    settings.need_comfy(); os.makedirs(os.path.join(CIN, "h3_sheet"), exist_ok=True)
    im.save(os.path.join(CIN, "h3_sheet", name + ".png"))
    return "h3_sheet/" + name + ".png"


def run_row(tag, rig, r, cols, seed, ref, calm=False, heads=None, lora=None, lora_strength=1.0, pixel=False):
    chibi = CHIBI.replace("big head, about 2.5 heads tall", f"head not too big, about {heads} heads tall") if heads else CHIBI
    desc = OWENS + ". " + (CALM if calm else "") + "In <Video 1> the subject is " + VIEWS[r] + " " + chibi
    if pixel:
        desc = desc.replace("flat colours, clean lineart", "pixel art, flat colours, clean dark outline")
    g = graph(ref, desc, seed, f"h3_sheet/{tag}_r{r}", lora=lora, lora_strength=lora_strength)
    seq = [k for k in cols for _ in range(R)]; seq += [seq[-1]] * (LENGTH - len(seq))
    for k, c in enumerate(seq):
        g[f"s{k}"] = {"class_type": "LoadImage", "inputs": {"image": upload(guide_frame(rig, r, c), f"{tag}_r{r}_{k:02d}")}}
    prev = ["s0", 0]
    for k in range(1, LENGTH):
        g[f"b{k}"] = {"class_type": "ImageBatch", "inputs": {"image1": prev, "image2": [f"s{k}", 0]}}
        prev = [f"b{k}", 0]
    g["60"]["inputs"]["image"] = prev
    secs, outs = wait(post(g))
    files = [os.path.join(COUT, i.get("subfolder", ""), i["filename"]) for i in outs["51"]["images"]]
    print(f"row {r}: {len(files)} frames, {secs:.0f} s", flush=True)
    return [Image.open(files[min(k * R + R // 2, len(files) - 1)]).convert("RGB") for k in range(len(cols))], files


def to_cell(im):
    small = kcentroid_down(im.resize((W, H), Image.LANCZOS), K, detail_bias=True)
    return small.crop((PAD, PAD, PAD + CW, PAD + CH))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--rig", default="rig_Men01_body")
    ap.add_argument("--rows", default="0,1,2,3"); ap.add_argument("--still", type=int, nargs=2); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--big", action="store_true", help="48x64 cells (a --grow 1.5 rig), 8x: 448x576")
    ap.add_argument("--k", type=int, help="upscale per sprite pixel (default 12, or 8 with --big)")
    ap.add_argument("--pad", type=int, default=4, help="grey sprite pixels around each cell")
    ap.add_argument("--calm", action="store_true", help="knee-length coat, neutral face, no talking or blinking")
    ap.add_argument("--heads", help="body height in heads for the prompt, e.g. 3 (default: big head, 2.5 heads)")
    ap.add_argument("--ref-head", type=float, default=1.0, help="shrink the reference's head, e.g. 0.8")
    ap.add_argument("--lora", help=r"extra LoRA, e.g. h3\MIniMax_TFMPixelOriginal_R2V.safetensors")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--pixel", action="store_true", help="ask for pixel art in the prompt")
    ap.add_argument("--prompt", help="prompt template file instead of prompts/h3_swap.txt ([add description here] is filled in)")
    ap.add_argument("--ref-sheet", help="a finished 48x64 _sheet.png: each row gets its own standing sprite (column 0), nearest-upscaled to the video grid, as <Picture 1>")
    ap.add_argument("--ref-image", help="use this image as <Picture 1> instead of the 4-view crop (e.g. a sprite nearest-upscaled to the video's grid)")
    a = ap.parse_args()
    global CW, CH, K, W, H, PAD, PROMPT
    if a.prompt:
        PROMPT = open(a.prompt, encoding="utf-8").read()
    if a.big:
        CW, CH, K = 48, 64, 8
    K, PAD = a.k or K, a.pad
    W, H = (CW + 2 * PAD) * K, (CH + 2 * PAD) * K
    if W % 32 or H % 32:
        raise SystemExit(f"{W}x{H} is not divisible by 32: change --k or --pad")
    print(f"video {W}x{H}, cell {CW}x{CH}, {K}x, pad {PAD}", flush=True)
    rig = Image.open(os.path.join(D, a.rig + "_1x.png")).convert("RGB")
    refim = Image.open(a.ref_image).convert("RGB") if a.ref_image else reference(a.ref_head); refim.save(os.path.join(D, a.tag + "_refimg.png")); ref = upload(refim, a.tag + "_ref")
    t = time.time()
    if a.still:
        r, c = a.still
        frames, files = run_row(a.tag, rig, r, [c] * 4, a.seed, ref, a.calm, a.heads, a.lora, a.lora_strength, a.pixel)
        frames[1].save(os.path.join(D, a.tag + "_h3raw.png"))
        guide_frame(rig, r, c).save(os.path.join(D, a.tag + "_guide.png"))
        to_cell(frames[1]).resize((CW * 8, CH * 8), Image.NEAREST).save(os.path.join(D, a.tag + "_x8.png"))
        print(a.tag, f"{time.time() - t:.0f} s"); return
    sheet = Image.new("RGB", (CW * 4, CH * 4), GREY)
    raw = Image.new("RGB", (W * 4 // 2, H * 4 // 2), GREY)
    if a.rows != "0,1,2,3" and os.path.exists(os.path.join(D, a.tag + "_1x.png")):
        # re-rolling some rows (e.g. --rows 3 --seed 2): keep the other rows of the existing sheet
        sheet.paste(Image.open(os.path.join(D, a.tag + "_1x.png")).convert("RGB"))
        if os.path.exists(os.path.join(D, a.tag + "_raw.png")):
            raw.paste(Image.open(os.path.join(D, a.tag + "_raw.png")).convert("RGB"))
    for r in [int(x) for x in a.rows.split(",")]:
        if a.ref_sheet:
            spr = Image.open(a.ref_sheet).convert("RGBA").crop((0, r * CH, CW, (r + 1) * CH))
            can = Image.new("RGBA", (CW + 2 * PAD, CH + 2 * PAD), GREY + (255,)); can.alpha_composite(spr, (PAD, PAD))
            can = can.convert("RGB").resize((W, H), Image.NEAREST); can.save(os.path.join(D, f"{a.tag}_refimg_r{r}.png"))
            ref = upload(can, f"{a.tag}_ref_r{r}")
        frames, _ = run_row(a.tag, rig, r, [0, 1, 2, 3], a.seed, ref, a.calm, a.heads, a.lora, a.lora_strength, a.pixel)
        for c, f in enumerate(frames):
            sheet.paste(to_cell(f), (c * CW, r * CH))
            raw.paste(f.resize((W // 2, H // 2), Image.LANCZOS), (c * W // 2, r * H // 2))
    sheet.save(os.path.join(D, a.tag + "_1x.png")); raw.save(os.path.join(D, a.tag + "_raw.png"))
    sheet.resize((CW * 16, CH * 16), Image.NEAREST).save(os.path.join(D, a.tag + "_x16.png"))
    json.dump({"layout": "64" if a.big else "48", "engine": "minimax_h3", "rig": a.rig, "seed": a.seed, "k": K, "pad": PAD, "video": [W, H], "calm": a.calm, "heads": a.heads, "ref_head": a.ref_head, "lora": a.lora, "lora_strength": a.lora_strength, "pixel": a.pixel, "ref_image": a.ref_image, "ref_sheet": a.ref_sheet}, open(os.path.join(D, a.tag + "_prompt.json"), "w"))
    print(a.tag, f"{time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
