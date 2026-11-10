"""Compose spec of Fig 3 with the added cohorts (eleven columns, so two rows): python -m analysis.tasks.F1_figures.compose --spec analysis/tasks/B2_benchmark/integration/fig_specs.py fig3_ext"""

from analysis.tasks.F1_figures.compose import Col, Figure, Panel, Row

FIGS = {"fig3_ext": Figure(Col(Row(Panel("fig3a_ext_integration_mixing", "a", ink=False), Panel("fig3b_ext_integration_biology", "b", ink=False), same_scale=True, gap=4),
                               Row(Panel("fig3c_ext_integration_composite", "c", ink=False), width=281)))}
