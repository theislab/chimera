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
    "main_tissues_3": {"words": "internal names on the panels; to change when they are redrawn"},
    "main_tissues_4": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    # specs/supp_deployment.py
    "supp_categories_broad": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size"},
    "supp_hallmark": {"text": "text under the floor; panels to be redrawn at print size"},
    # specs/supp_protein.py
    "supp_residue_audit": dict.fromkeys(["text", "text_single", "scale", "unreadable_text_layer"],
                          "holds a region of a page assembled by hand; its panels are to be redrawn at print size"),
    "supp_residue_audit_2": {"text": "holds regions of a page assembled by hand; its panels are to be redrawn at print size"},
    # specs/supp_tissues.py
    # specs/supp_deployment.py
    "supp_known_measures": {"text": "text under the floor; panels to be redrawn at print size"},
    "supp_functional_layer": {"letters": "letters to be put in reading order together with the caption"},
    # specs/supp_model.py
    "supp_integration_ten_methods": {"spelling": "British spelling on the panels; to change when they are redrawn"},
    # specs/supp_deployment.py
    "supp_celltype_seeds": {"spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_relations_heatmap": {"words": "internal names on the panels; to change when they are redrawn"},
    "supp_relations_density": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size",
        "words": "internal names on the panels; to change when they are redrawn"},
    # specs/supp_validation.py
    "supp_gate_gap": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size",
        "words": "internal names on the panels; to change when they are redrawn",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_gate_gap_classes": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size",
        "words": "internal names on the panels; to change when they are redrawn",
        "spelling": "British spelling on the panels; to change when they are redrawn",
        "letters": "letters to be put in reading order together with the caption"},
    "supp_gate_gap_named": {
        "spelling": "British spelling on the panels; to change when they are redrawn",
        "letters": "letters to be put in reading order together with the caption"},
    # specs/supp_tissues.py
    "supp_programs_recurrent": {
        "text": "text under the floor; panels to be redrawn at print size",
        "words": "internal names on the panels; to change when they are redrawn",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_programs_by_tissue": {
        "text": "text under the floor; panels to be redrawn at print size",
        "words": "internal names on the panels; to change when they are redrawn",
        "letters": "letters to be put in reading order together with the caption"},
    "supp_programs_named_genes": {
        "text": "text under the floor; panels to be redrawn at print size",
        "letters": "letters to be put in reading order together with the caption"},
    # specs/supp_protein.py
    "supp_protein_architecture": {
        "text": "text under the floor; panels to be redrawn at print size",
        "words": "internal names on the panels; to change when they are redrawn",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    # specs/supp_tissues.py
    "supp_cell_compartments": {"spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_cell_lineages": {"spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_primate_branches": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size",
        "words": "internal names on the panels; to change when they are redrawn"},
    "supp_disease_genes": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "shrunk as a whole, as it was in LaTeX; to be laid out at full width",
        "words": "internal names on the panels; to change when they are redrawn",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_drug_targets": {"letters": "letters to be put in reading order together with the caption"},
    # specs/supp_protein.py
    "supp_regulatory_architecture": {"spelling": "British spelling on the panels; to change when they are redrawn"},
    # specs/supp_validation.py
    "supp_published_differences": {
        "text": "text under the floor; panels to be redrawn at print size",
        "text_single": "single characters under the floor; panels to be redrawn at print size",
        "scale": "shrunk as a whole, as it was in LaTeX; to be laid out at full width"},
    # specs/supp_model.py
    "supp_integration_eleven_cohorts": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "shrunk as a whole, as it was in LaTeX; to be laid out at full width"},
    # specs/supp_validation.py
}
