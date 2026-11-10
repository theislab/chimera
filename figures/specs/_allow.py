"""Figures that do not pass the house style yet, with the reason. The build prints this list every time.

key -> {check: reason}. Checks: text, text_single, fonts, font_family, page, width, scale, edge, words, spelling,
letters, unreadable_text_layer. An entry whose check passes is an error, so the list can only shrink. The floor of
5.0 pt is binding for the main figures: their entries are work in progress, not exceptions.
"""

ALLOW: dict = {
    # specs/main.py
    "main_alpha_vs_prior": dict.fromkeys(["text", "text_single", "font_family", "scale"],
                           "page assembled by hand, placed as it printed; to be rebuilt from its panels"),
    "main_delta_alpha_liver": dict.fromkeys(["text", "text_single", "scale"],
                              "holds a region of a page assembled by hand; its panels are to be redrawn at print size"),
    "main_delta_alpha_liver_2": {
        "text": "text under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn",
        "letters": "letters to be put in reading order together with the caption"},
    "main_validation_2": {
        "text": "text under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "main_tissues": {"spelling": "British spelling on the panels; to change when they are redrawn"},
    "main_tissues_2": {"text": "text under the floor; panels to be redrawn at print size"},
    "main_tissues_3": {
        "text": "text under the floor; panels to be redrawn at print size",
        "text_single": "single characters under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size",
        "words": "internal names on the panels; to change when they are redrawn"},
    "main_tissues_4": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "main_protein": dict.fromkeys(["text", "scale"],
                    "holds a region of a page assembled by hand; its panels are to be redrawn at print size"),
    "main_protein_2": dict.fromkeys(["text", "text_single", "font_family", "scale", "unreadable_text_layer"],
                      "holds a region of a page assembled by hand; its panels are to be redrawn at print size"),
    # specs/supp_deployment.py
    "supp_categories_broad": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size"},
    "supp_categories_testis": {"spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_hallmark": {"text": "text under the floor; panels to be redrawn at print size"},
    # specs/supp_protein.py
    "supp_residue_audit": dict.fromkeys(["text", "text_single", "scale"],
                          "holds a region of a page assembled by hand; its panels are to be redrawn at print size"),
    # specs/supp_tissues.py
    "supp_tissue_bone_marrow": {
        "text": "text under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_tissue_kidney": {
        "text": "text under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_tissue_islet": {
        "text": "text under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_tissue_eye": {
        "text": "text under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_tissue_cortex": {
        "text": "text under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_tissue_lung": {
        "text": "text under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_tissue_muscle": {
        "text": "text under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_tissue_tumor_myeloid": {
        "text": "text under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_tissues_recurrence": {"text": "text under the floor; panels to be redrawn at print size"},
    "supp_tissues_lineage": {"spelling": "British spelling on the panels; to change when they are redrawn"},
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
    "supp_benchmark_overview": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size",
        "words": "internal names on the panels; to change when they are redrawn",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_benchmark_tasks": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "panels placed below 0.9 of their drawn size; to be redrawn at print size",
        "words": "internal names on the panels; to change when they are redrawn"},
    # specs/supp_model.py
    "supp_integration_eleven_cohorts": {
        "text": "text under the floor; panels to be redrawn at print size",
        "scale": "shrunk as a whole, as it was in LaTeX; to be laid out at full width"},
    # specs/supp_validation.py
    "supp_ground_truth": {
        "text": "text under the floor; panels to be redrawn at print size",
        "text_single": "single characters under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_ground_truth_2": {
        "text": "text under the floor; panels to be redrawn at print size",
        "text_single": "single characters under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_ground_truth_3": {
        "text": "text under the floor; panels to be redrawn at print size",
        "text_single": "single characters under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
    "supp_ground_truth_4": {
        "text": "text under the floor; panels to be redrawn at print size",
        "text_single": "single characters under the floor; panels to be redrawn at print size",
        "spelling": "British spelling on the panels; to change when they are redrawn"},
}
