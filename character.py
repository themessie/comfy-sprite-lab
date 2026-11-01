"""The character being animated. Replace these for your own character: a one-line description for the prompts, and a design
sheet with the front view (the H3 reference in round one; its colours also feed sheet_cleanup.py --design)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "owens"
DESCRIPTION = ("a chibi bearded detective with messy dark brown hair, wearing a knee-length beige trench coat, white shirt, navy tie, "
               "dark brown trousers and brown shoes")
DESIGN_SHEET = os.path.join(HERE, "examples", "owens", "owens_4view.png")
FRONT_CROP = (50, 80, 340, 610)        # the front view's box on the design sheet
NECK = 235                             # neck line inside FRONT_CROP (for --ref-head: shrink everything above it)
