# sprite-forge

Turn a character design into a **48×64, 4-direction RPG walk sheet that is real pixel art**: every pixel on the grid, the same face
and outfit in every frame, and the motion of a proper walk cycle. Built on ComfyUI, with two routes:

| Route | Models | Results |
|---|---|---|
| **Wan route** (shown here) | Qwen-Image 2.1 + Wan 2.2 (Apache-2.0) | Good, and shareable anywhere |
| **H3 route** | MiniMax H3 + SD 1.5 + 8bitdiffuser | Better: crisper, exact poses. But its licence bars showing H3 outputs in the EU, UK, South Korea and USA, so **no H3 results are published here** |

> All example images in this repository are **AI-generated** (Qwen-Image 2.1, Wan 2.2), then post-processed by this pipeline.
> Using the pipeline means using these models under their own licences: see [Licences and credits](#licences-and-credits).

| Design (simple style) | Walk sheet | Walking |
|---|---|---|
| <img src="examples/mara/mara_design.png" width="380"> | <img src="examples/mara/mara_walk_sheet.png" width="230"> | <img src="examples/mara/mara_walk.gif" width="300"> |

*Mara, a newspaper seller, made entirely with the Wan route (front, left, right, back; 48×64 cells).*

## The Wan route

| # | Step | Script | What it does |
|---|---|---|---|
| 0 | Design sheet | `make_design_sheet.py --style simple` | Qwen-Image 2.1 dresses the base body's four standing views as your character: the same person from front, right, back and left. **Simple style**: thick dark outlines, flat colours, no small details, so it survives the shrink to 48×64. See [The design sheet](#the-design-sheet). |
| 1 | Walk loops | `gen_wan_walk.py` | Per direction, Wan 2.2 image-to-video walks in place from that view and back to it (a loop). The gait is read from the leg spread; the most closed pose and the two strides become the 4 frames. Front and back walks get a high-knee prompt (they barely move the legs otherwise). |
| 2 | Downscale ÷8 | (inside step 1) | Frames are made at exactly 8× the sprite (512×640 for 48×64 + 8 px margin); k-centroid per 8×8 block with true colours. |
| 3 | Sheet post | `sheet_post.py --stand --colors 32` | Background removed, one 32-colour palette for the sheet, standing frame copied to column 2. |
| 4 | Clean-up rules | `sheet_cleanup.py --thin ...` | Doubled outlines thinned to 1 px, specks merged, spikes/notches fixed, **head lock** (the standing head copied into the stride frames), selective outline. |
| 5 | Pixel fixes (optional) | `sheet_to_pxl.py` + [pxl](https://github.com/crsolver/claude-pxl) | The sheet becomes a text file (pixel-exact); fix single pixels as code and export strips, GIFs, Aseprite-style anim JSON and a TexturePacker sheet. |
| R2 | Round two (optional) | `gen_wan_walk.py --start-sheet SHEET` | Wan starts from the finished standing sprites (×8) instead of the design, for further takes of the same character. |

```bash
export SPRITE_FORGE_CHARACTER=mara                            # characters/mara.json
python make_design_sheet.py Women01 --style simple --seeds 1,2,3,4   # pick one, set design_sheet in characters/mara.json
python build_rig_sheet.py Men01 --grow 1.5                    # guide geometry (figure size, foot line) for step 1
python gen_wan_walk.py mara_walk                              # 4 directions, about 3 min each on a 12 GB card
python sheet_post.py mara_walk --stand --colors 32 --out mara_walk_rpg
python sheet_cleanup.py mara_walk_rpg mara_final --thin --orphans --silhouette --headlock --selout
```

### What the tests showed (Wan route)

- **The design decides most of it.** The same pipeline on a detailed design (freckles, knit texture, buttons) gives muddy faces at
  48×64; a design drawn for a tiny sprite (thick outlines, flat colours, big shapes) gives clean, readable frames.
  ([comparison](examples/mara/cmp_detailed_vs_simple.png))
- **Direct ÷8 beats every pixel-pass variant** on Wan frames of a simple design: the SD 1.5 + 8bitdiffuser pass that the H3 route
  needs only adds noise here (at 0.35 and more so at 0.45), and starting Wan from a pixelated image or adding the Wan 2.1 Pixel
  Toolkit LoRA does not help either. ([12 variants](examples/mara/cmp_route_variants.png))
- **k-centroid darkens**: a block half outline, half fill goes wholly to the winner, so outlines and dark areas grow; the colour
  rounding (`//8`) adds a little. True colours plus outline thinning undo most of it.
  ([comparison](examples/mara/cmp_downscale_darkening.png))
- Wan keeps the facing direction in all four views (H3 sometimes turns the back view around), and its side strides are lively; its
  front and back walks move the legs little unless asked for high knees.

## The H3 route

Higher quality where you may use it: MiniMax H3 swaps your character onto an RPG Maker base-body walk, so poses follow the guide
exactly; an 8bitdiffuser pixel pass puts the frames on a true pixel grid.

| # | Step | Script | What it does |
|---|---|---|---|
| 1 | Guide walk | `build_rig_sheet.py` | Your RPG Maker XP style base-body walk (4×4 frames), grown to 48×64 cells. |
| 2 | Character swap | `gen_h3_sheet.py` | MiniMax H3 per direction: the guide row, nearest-upscaled 6× and each pose held 5 frames, is the motion (`<Video 1>`); the design's front view with the head shrunk to 80 % is the character (`<Picture 1>`). |
| 3 | Outline expansion | `gen_px64_pass.py --oe 3` | Each frame at 8× (512×640); PixelOE thickens small high-contrast details so they survive. |
| 4 | Pixel pass | `gen_px64_pass.py --mode sheet` | All 16 frames as one 2048×2560 image, redrawn by SD 1.5 + the 8bitdiffuser 64x LoRA at 0.45 (one flat pixel per 8×8 latent cell), then ÷8. `--align row` shifts each row to the grid window that keeps the most outline ink. |
| 5–7 | Post, clean-up, pixel fixes | as in the Wan route | |
| R2 | Round two | `gen_h3_sheet.py --k 8 --pad 8 --ref-sheet SHEET` | H3 at 8× with each row's standing sprite as its reference copies the pixel blocks, so ÷8 is direct and steps 3–4 drop out. Fixes made in pxl carry into every new frame. |

```bash
python build_rig_sheet.py Men01 --grow 1.5
python gen_h3_sheet.py owens --calm --big --rig rig_Men01_body_big --k 6 --pad 8 --ref-head 0.8
python gen_px64_pass.py owens_px --src owens --denoise 0.45 --mode sheet --oe 3 --align row
python sheet_post.py owens_px --stand --colors 32 --out owens_px_rpg
python sheet_cleanup.py owens_px_rpg owens_final --orphans --silhouette --headlock --selout
python gen_h3_sheet.py owens_r2 --calm --big --rig rig_Men01_body_big --k 8 --pad 8 --ref-sheet owens_final_sheet.png   # round two
```

What the tests showed: H3 keeps a character consistent and follows the guide pose exactly. It draws smooth lines with no pixel grid
of its own (step corners cluster on no block size from 2 to 12 px), so no downscale window makes it lossless; the 8bitdiffuser pass
imposes the grid, best when the whole sheet is redrawn as one image. At 8× with a pixel-art reference per direction, H3 copies the
pixel blocks (its 32×32-pixel latent patches hold exactly 4×4 sprite pixels at 8×). **Check the back row:** H3 sometimes turns the
back view to face the camera in later frames; re-roll that row with `--rows 3 --seed 2`. The prompt template is
`prompts/h3_swap.txt` (`--prompt FILE` for another).

## The design sheet

The front view is what matters most (it is the H3 reference, and both routes take its colours); the side and back views are the Wan
route's start images. A good design for a 48×64 sprite: the whole body from head to shoes, chibi proportions, **thick dark
outlines, flat colours, big simple shapes, no small details**.

```bash
python make_design_sheet.py Women01 --style simple --seeds 1,2,3,4   # -> work/design_Women01_simple_s<seed>.png, each measured
python make_design_sheet.py --measure my_drawing.png                 # measure a design sheet you drew yourself
```

It puts the base body's four standing frames side by side as a mannequin, and Qwen-Image 2.1 (edit, Turbo 8-step LoRA) dresses it as
the character's description in one call. Seeds differ a lot: make a few and pick one
([detailed](examples/mara/cmp_detailed_designs.png) vs [simple](examples/mara/cmp_simple_designs.png) style).

**Characters** live in `characters/<name>.json`: `description` (one line, used in every prompt), `design_sheet`, and optionally
`front_crop` and `neck` (measured from the sheet when left out). Pick one with `"character"` in `settings.json` or
`SPRITE_FORGE_CHARACTER`.

## Requirements

- An NVIDIA GPU (developed on an RTX 3080 Ti, 12 GB) and **ComfyUI** with
  [Kijai's WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Wan route) and/or the MiniMax H3 and LTX-Video nodes
  (H3 route).
- Models, under the file names the graphs use:

  | Model | File(s) | Route | Licence |
  |---|---|---|---|
  | Qwen-Image 2.1 + Turbo 8-step LoRA | `qwen_image_2.1_int8_convrot`, `qwen3vl_8b_int8_convrot`, `qwen_image_2.1_vae_bf16`, `Qwen-Image-2.1-Turbo8-8step-lora` | both (design) | Apache-2.0 |
  | Wan 2.2 I2V A14B (GGUF) | `Wan2.2-I2V-A14B-High/LowNoise-Q5_K_S.gguf`, `wan_2.1_vae`, `umt5-xxl-enc-fp8_e4m3fn` | Wan | Apache-2.0 |
  | lightx2v 4-step LoRAs | `wan2.2_i2v_A14b_high/low_noise_lora_rank64_lightx2v_4step_1022` | Wan | see the model page |
  | MiniMax H3 Ref2VA + LoRAs | `minimax_h3_ref2va_pruned_int8_convrot`, `qwen3vl_32b_minimax_h3_nvfp4_awq`, H3 VAEs, `h3/minimax_h3_ref2v_turbo_4step_v0.1`, `h3/minimax_h3_character_swap_v1.1` | H3 | MiniMax H3 Community License |
  | Stable Diffusion 1.5 + 8bitdiffuser 64x v4 | `v1-5-pruned.ckpt`, `PX64NOCAP_epoch_10` ([Civitai 185743](https://civitai.com/models/185743)) | H3 (pixel pass) | CreativeML Open RAIL-M |

- Python 3.11+ with `pip install -r requirements.txt`.
- For pixel fixes: [pxl](https://github.com/crsolver/claude-pxl) (MIT) and Node ≥ 22.18.
- **A base-body walk sheet** in `base_bodies/` (RPG Maker XP layout: 4×4 frames, rows down/left/right/up), used as the design
  mannequin and the H3 guide. Not included: use one you are licensed to use.

Copy `settings.example.json` to `settings.json` and set the ComfyUI input/output folders (and Node/pxl paths for pixel fixes).

## Status

Both routes were run end to end from this repository (H3 on 2026-11-01, Wan on 2026-11-03). Other tools: `gen_pixeloe.py` (PixelOE
as the downscaler) and `ink_align.py` (direct ÷6 at the window with the most outline ink).

## Licences and credits

- **Code:** MIT (`LICENSE`), except the detail-bias function in `pixelops.py`, adapted from
  [sprite-gen](https://github.com/aldegad/sprite-gen) (Copyright 2026 Alex Kim, Apache-2.0, `LICENSES/Apache-2.0.txt`). See `NOTICE`.
- **Credits:** k-centroid downscaling after Astropulse (Retro Diffusion); outline expansion by
  [PixelOE](https://github.com/KohakuBlueleaf/PixelOE) (KohakuBlueleaf); pixel fixes and exports with
  [pxl](https://github.com/crsolver/claude-pxl) (crsolver); Wan runs through Kijai's WanVideoWrapper; everything runs in
  [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Full list with licences: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Models:** none are included; each has its own licence. **MiniMax H3** (Community License Agreement, H3 route only): use and
  distribution only outside the EU, UK, South Korea and USA (outputs included); outputs may not be used to train other AI models; a
  commercial product or service using it must show "MiniMax H3" prominently in its UI; AI-generated content shared publicly must be
  disclosed as such. MiniMax H3 is licensed under the MiniMax H3 Community License Agreement. Copyright © 2026 MiniMax. All Rights
  Reserved.
- **Examples:** the Owens and Mara characters and all images in `examples/` are © themessie, all rights reserved, and AI-generated.
- **Base bodies** are not included; use sprites you have the rights to.

This is a summary, not legal advice; the licence texts linked in THIRD_PARTY_NOTICES.md are what count.
