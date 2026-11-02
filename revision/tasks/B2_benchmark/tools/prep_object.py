"""Write one setting's object for the added tools (chimera_env).  python -m ...tools.prep_object <cohort> <setting> <rep> <out.h5ad>"""
from __future__ import annotations

import sys

from revision.tasks.B2_benchmark import perturb as P


def main():
    c, s, r, out = sys.argv[1:5]
    P.build(dict(cohort=c, setting=s, rep=int(r)), out_path=out, check=False)


if __name__ == "__main__":
    main()
