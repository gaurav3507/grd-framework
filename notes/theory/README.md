# GRD theory notes

`grd_theory_note.tex` / `.pdf` with `theory.bib`: validity and consistency (5 Oct 2026, revised 8 Oct 2026 for BY, the c(m) factor, and the pooled Norman B=19,999 result). Superseded by Section 3.5 and Appendix D of the manuscript.
The numerical checks in `checks/` were run in a sandbox (numpy 2.5.3) against `src/gate/precision_readout.py` at commit c9ebb9e.
The check scripts reference sandbox paths; they are kept for provenance, not as runnable experiments.
`lower_bound/`: lower-bound note (Theorems 9 and 10, Corollary 11 of the manuscript), 7 Oct 2026; `lb_check.py` ran in a sandbox (numpy 2.5.3) against `src/gate/precision_readout.py` at d7a322f; `lb_check.json` is its output.
`checks/e8_validity_check.txt`: output of `e8_pooled_null.py --validity-check` on this Mac.
