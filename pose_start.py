"""Posed start images: the character redrawn by Qwen-Image 2.1 in the pose of an OpenPose skeleton, as Wan's start (and end) frame.

Wan copies the start pose: from a neutral standing frame, front and back walks shuffle (the feet barely lift) unless the wording and
the seed are lucky. Starting mid-step, Wan has to complete a real step cycle to get back to that frame. The idea of posing the
keyframe from a skeleton comes from mor-o/comfyui-2d-character-pipeline (workflow 1, credited in THIRD_PARTY_NOTICES.md); here the
skeleton is the second image of Qwen 2.1's own multi-image edit, so no ControlNet is needed.

Skeletons are COCO-18 keypoints in figure units (x from the figure's centre, y from the top of the cap, both / figure height),
measured on Mara's front view, so they fit chibi proportions; back views swap left and right and drop the face points.

Usage: python pose_start.py --dir down --pose walk_step --seeds 2,3
       python forge.py walk --dirs down --start down=work/forge/poses/mara_walk_step_down_s3.png
"""
import argparse, os, shutil
import numpy as np
from PIL import Image, ImageDraw
import character, settings
from comfy import post, wait, CIN, COUT
from forge import VIEW, FACING, cut_view, start_image
from gen_wan_walk import BACKGROUNDS, magentaish
from make_design_sheet import qwen_graph

# keypoints in reference pixels of Mara's 512x640 front start image (figure centre x 255, top y 130, height 429); index order
# 0 nose, 1 neck, 2-4 right shoulder/elbow/wrist, 5-7 left, 8-10 right hip/knee/ankle, 11-13 left, 14/15 right/left eye, 16/17 ears.
# The person's right is the image's left in a front view.
REF = (255, 130, 429)
POSES = {
    "stand": ([(256, 252), (256, 282), (212, 286), (192, 342), (180, 402), (300, 286), (320, 342), (332, 402), (234, 395), (230, 465),
               (228, 528), (278, 395), (282, 465), (284, 528), (232, 232), (280, 232), (206, 240), (306, 240)],
              "standing relaxed, arms at the sides"),
    "walk_step": ([(256, 252), (256, 282), (212, 286), (196, 338), (196, 396), (300, 286), (322, 344), (338, 404), (234, 395), (229, 466),
                   (226, 530), (278, 395), (286, 448), (290, 492), (232, 232), (280, 232), (206, 240), (306, 240)],
                  "walking, mid-step: one knee raised and that foot lifted off the ground, the other leg straight with the foot on the "
                  "ground, arms relaxed at the sides"),
    "run_stride": ([(256, 254), (256, 284), (212, 288), (200, 345), (214, 374), (300, 288), (314, 345), (300, 378), (234, 397),
                    (228, 470), (224, 526), (278, 397), (288, 440), (294, 478), (232, 234), (280, 234), (206, 242), (306, 242)],
                   "running, mid-stride: one knee high and that foot well off the ground, the other leg pushing off, arms bent at the "
                   "elbows at the sides of the body, hands at waist height, the hands never cross in front of the chest"),
}
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
         (0, 15), (15, 17)]
COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170),
           (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 85), (255, 0, 0)]
SWAP = {2: 5, 3: 6, 4: 7, 8: 11, 9: 12, 10: 13, 14: 15, 16: 17}


def keypoints(pose, direction, box):
    """The pose's keypoints placed on a figure box (centre x, top y, height); None where the view hides a point."""
    cx, top, h = REF
    pts = [((x - cx) / h, (y - top) / h) for x, y in POSES[pose][0]]
    if direction == "up":                                 # seen from behind: left and right change sides, no face
        sw = {**SWAP, **{v: k for k, v in SWAP.items()}}
        pts = [pts[sw.get(i, i)] for i in range(18)]
        for i in (0, 14, 15):
            pts[i] = None
    bx, by, bh = box
    return [None if p is None else (bx + p[0] * bh, by + p[1] * bh) for p in pts]


def draw(kp, size, w=8):
    im = Image.new("RGB", size); d = ImageDraw.Draw(im)
    for i, (a, b) in enumerate(LIMBS):
        if kp[a] and kp[b]:
            d.line((*kp[a], *kp[b]), fill=tuple(int(c * 0.6) for c in COLOURS[i]), width=w)
    for i, p in enumerate(kp):
        if p:
            d.ellipse((p[0] - w * 0.6, p[1] - w * 0.6, p[0] + w * 0.6, p[1] + w * 0.6), fill=COLOURS[i])
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="down", choices=["down", "up"])
    ap.add_argument("--pose", default="walk_step", choices=sorted(POSES)); ap.add_argument("--seeds", default="2,3")
    ap.add_argument("--cell", default="48x64"); ap.add_argument("--margin", type=int, default=8); ap.add_argument("--ref-head", type=float, default=0.8)
    a = ap.parse_args()
    cw, ch = (int(v) for v in a.cell.lower().split("x")); bgc, bg_words = BACKGROUNDS["magenta"]
    start = start_image(cut_view(character.DESIGN_SHEET, VIEW[a.dir][1], a.ref_head), cw, ch, a.margin, bgc)
    ys, xs = np.nonzero(~magentaish(np.asarray(start)))
    box = ((xs.min() + xs.max()) / 2, ys.min(), ys.max() - ys.min())
    out = os.path.join(settings.WORK, "forge", "poses"); os.makedirs(out, exist_ok=True)
    name = f"{character.NAME}_{a.pose}_{a.dir}"
    skel = draw(keypoints(a.pose, a.dir, box), start.size)
    skel.save(os.path.join(out, name + "_skeleton.png")); Image.blend(start, skel, 0.5).save(os.path.join(out, name + "_overlay.png"))
    os.makedirs(os.path.join(CIN, "sprite_forge"), exist_ok=True)
    start.save(os.path.join(CIN, "sprite_forge", name + "_src.png")); skel.save(os.path.join(CIN, "sprite_forge", name + "_skel.png"))
    prompt = (f"Image 1 shows a character; image 2 is an OpenPose skeleton. Redraw the character from image 1 ({character.DESCRIPTION}) "
              f"exactly: same art style, outfit, colours, size and position, {FACING[a.dir]}, on the same flat {bg_words} background, but "
              f"in the pose of the skeleton in image 2: {POSES[a.pose][1]}."
              + (" We see the back of the clothes: no buttons, no opening and no shirt front, the back stays plain as in image 1."
                 if a.dir == "up" else ""))      # without this, Qwen often draws the jacket's front (buttons, blouse) on the back
    for seed in (int(s) for s in a.seeds.split(",")):
        g = qwen_graph(f"sprite_forge/{name}_src.png", prompt, seed, f"sprite_forge/poses/{name}_s{seed}")
        g["11"] = {"class_type": "LoadImage", "inputs": {"image": f"sprite_forge/{name}_skel.png"}}
        g["25"]["inputs"]["images.image_2"] = ["11", 0]
        t, o = wait(post(g)); i = o["51"]["images"][0]
        dst = os.path.join(out, f"{name}_s{seed}.png"); shutil.copy(os.path.join(COUT, i.get("subfolder", ""), i["filename"]), dst)
        print(f"{dst} ({t:.0f} s)   ->  forge.py ... --dirs {a.dir} --start {a.dir}={dst}", flush=True)


if __name__ == "__main__":
    main()
