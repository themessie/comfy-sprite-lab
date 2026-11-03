# Third-party notices and credits

sprite-forge's own code is MIT-licensed (`LICENSE`). It bundles **one** piece of third-party code; everything else below is
software or a model you install yourself. Each keeps its own licence, and those terms apply to your use of it and of what it
generates. This file is a summary to help you find them, not legal advice; the linked licence texts are what count.

## Code included in this repository

| Component | Where | Licence | Notes |
|---|---|---|---|
| sprite-gen, `_dominant_block_color` detail bias | `pixelops.py` (`_dark_minority`) | Apache-2.0, Copyright 2026 Alex Kim ([aldegad/sprite-gen](https://github.com/aldegad/sprite-gen)) | Adapted and modified (see the function's docstring). Licence text: `LICENSES/Apache-2.0.txt`. |

## Methods and ideas (no code copied)

| What | Credit |
|---|---|
| k-centroid downscaling (`pixelops.kcentroid_down`) | Astropulse's K-Centroid method (Retro Diffusion); our own implementation |
| Selective outline (`sheet_cleanup.py --selout`) | A common pixel-art technique; the "darker shade of the neighbouring fill" rule as described in pixel-art tutorials (e.g. Lospec) |
| Contrast-aware outline expansion | Used through the PixelOE package (below) |

## Software you install (not included)

| Software | Licence | Used for |
|---|---|---|
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | GPL-3.0 | Runs all models; sprite-forge only talks to it over its HTTP API |
| [PixelOE](https://github.com/KohakuBlueleaf/PixelOE) (KohakuBlueleaf, Shin-Ying Yeh) | Apache-2.0 | Outline expansion before the pixel pass (`gen_px64_pass.py --oe`, `gen_pixeloe.py`) |
| [pxl](https://github.com/crsolver/claude-pxl) (crsolver) | MIT | Pixel fixes as code and the exports in `examples/owens/pxl/out/` |
| NumPy, Pillow, SciPy, PyTorch | BSD-style / HPND / BSD / BSD-style | Python dependencies |

## Models (not included)

| Model | Licence | What it means for you |
|---|---|---|
| [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (Ref2VA, text encoder, VAEs) | [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) (2 Aug 2026) with an Acceptable Use Policy (Exhibit A) | Use and distribution only within the Applicable Territory: worldwide **except the EU, UK, South Korea and USA**, and outputs may not be distributed or displayed outside it. Outputs may not be used to train or improve other AI models. A commercial product or service using it must show "MiniMax H3" prominently in its UI, and needs written authorisation above US$20 M yearly revenue. AI-generated content shared publicly must be disclosed as such. Its text encoder builds on Qwen3-VL (Apache-2.0). |
| H3 LoRAs: `minimax_h3_ref2v_turbo_4step`, `minimax_h3_character_swap_v1.1` | See their model pages; as modifications of H3 they also fall under the H3 licence | |
| Qwen-Image 2.1 + Turbo 8-step LoRA, Qwen3-VL encoder (`make_design_sheet.py`) | Apache-2.0 (check each model page) | |
| Stable Diffusion 1.5 (`v1-5-pruned.ckpt`) | CreativeML Open RAIL-M | Use-based restrictions (Attachment A of the licence) apply to you and pass on to anyone you share the model with |
| [8bitdiffuser 64x](https://civitai.com/models/185743) v4 (`PX64NOCAP_epoch_10`) | CreativeML Open RAIL-M + Civitai permissions | Civitai lists: use without credit, sell images, use commercially allowed |

## Inputs you bring

- **Base-body walk sheets** (`base_bodies/`): not included. Use sprites you have the rights to; many RPG Maker templates may only be
  used with RPG Maker or may not be redistributed, and guide sheets made from them are derived from them.
- **Your character design**: yours.

## Example images

Everything in `examples/owens/` is **AI-generated** (MiniMax H3, Qwen-Image 2.1, Stable Diffusion 1.5 with the 8bitdiffuser LoRA),
then post-processed by this pipeline and hand-fixed with pxl. The Owens character and these images are © themessie, all rights
reserved; they are shown as examples and are not covered by the MIT licence.
