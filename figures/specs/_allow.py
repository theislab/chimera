"""Figures that do not pass the house style yet, with the reason. The build prints this list every time.

key -> {check: reason}. Checks: text, text_single, fonts, font_family, page, width, scale, edge, words, spelling,
letters, unreadable_text_layer. An entry whose check passes is an error, so the list can only shrink. The floor of
5.0 pt is binding for every figure.
"""

ALLOW: dict = {
    # specs/main.py
    "main_protein": {
        "scale": "the atlas is composed 422 bp wide, its panels at 0.86 to 0.87 of their drawn size, so that the "
                 "figure is at most 242 bp high; the smallest text prints at 5.0 pt"},
    # specs/supp_protein.py
    "supp_atlas_archetypes": {
        "scale": "panel b is the panel stored by the notebook (it needs the trained gate network); it is drawn for a "
                 "larger page and placed at 0.775, where its smallest text prints at 5.0 pt"},
}
