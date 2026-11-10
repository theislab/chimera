from analysis.tasks.F1_figures.compose import Col, Figure, Panel, Row

P = Panel
FIGS = {"fig_c1_new_use_cases": Figure(Col(Row(P("c1a_near_identical", "a"), P("c1b_heldout_regions", "b")),
                                           Row(P("c1c_transfer", "c"), P("c1d_tumour_mixing", "d"), P("c1e_embedding", "e"))))}
