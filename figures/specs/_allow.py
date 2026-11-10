"""Figures that do not pass the house style yet, with the reason. The build prints this list every time.

key -> {check: reason}. Checks: text, text_single, fonts, font_family, page, width, scale, edge, words, spelling,
letters, unreadable_text_layer. An entry whose check passes is an error, so the list can only shrink. The floor of
5.0 pt is binding for the main figures: their entries are work in progress, not exceptions.
"""

ALLOW: dict = {
    # specs/main.py
    "main_protein_3": {
        "scale": "panel j is the panel stored by the notebook (it needs the trained gate network); it is drawn for a "
                 "larger page and placed at 0.775, where its smallest text prints at 5.0 pt"},
    # specs/supp_deployment.py
    "supp_hallmark": {"text": "text under the floor; panels to be redrawn at print size"},
    # specs/supp_protein.py
    "supp_residue_audit_2": {"text": "holds regions of a page assembled by hand; its panels are to be redrawn at print size"},
    # specs/supp_tissues.py
    # specs/supp_deployment.py
    # specs/supp_model.py
    # specs/supp_deployment.py
    "supp_celltype_seeds": {"spelling": "British spelling on the panels; to change when they are redrawn"},
    # specs/supp_validation.py
    # specs/supp_tissues.py
    # specs/supp_protein.py
    # specs/supp_tissues.py
    # specs/supp_protein.py
    # specs/supp_validation.py
    # specs/supp_model.py
    "supp_integration_eleven_cohorts": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "shrunk as a whole, as it was in LaTeX; to be laid out at full width"},
    # specs/supp_validation.py
}
