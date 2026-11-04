"""The character being animated, from characters/<name>.json: a one-line description for the prompts, a design sheet whose front
view is the H3 reference in round one (its colours also feed sheet_cleanup.py --design), the front view's box on that sheet and
the neck line inside it (for --ref-head); optionally design_diagonals, a sheet of four three-quarter views for 8-direction
animations (make_design_sheet.py --diagonals). Pick the character with "character" in settings.json or SPRITE_FORGE_CHARACTER
(default: owens). front_crop / neck may be null: they are then measured from the design sheet (make_design_sheet.py --measure
prints them, to pin them down by hand)."""
import json, os
import settings

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = os.environ.get("SPRITE_FORGE_CHARACTER") or settings._cfg.get("character") or "owens"
_c = json.load(open(os.path.join(HERE, "characters", NAME + ".json"), encoding="utf-8"))
DESCRIPTION = _c["description"]
DESIGN_SHEET = os.path.join(HERE, _c["design_sheet"])
DESIGN_DIAGONALS = os.path.join(HERE, _c["design_diagonals"]) if _c.get("design_diagonals") else None   # front-left, front-right, back-left, back-right


def _measured():
    from make_design_sheet import measure
    _, crop, neck = measure(DESIGN_SHEET)
    return crop, neck


if _c.get("front_crop") and _c.get("neck"):
    FRONT_CROP, NECK = tuple(_c["front_crop"]), _c["neck"]
elif os.path.exists(DESIGN_SHEET):
    FRONT_CROP, NECK = _measured()
else:
    FRONT_CROP, NECK = None, None      # no design sheet yet: make one with make_design_sheet.py
