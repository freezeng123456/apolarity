# Four-hidden-layer replication

User authorized a new four-layer network run after the three-hidden-layer pilot. Four layers here explicitly means four tanh hidden layers of width 128, followed by the scalar linear output. The MLP implementation counts affine layers, so pass depth=5. The architecture is [input,128,128,128,128,1].

Invoke run_fixed_wall_matrix.py --hidden-layers 4 --out <fresh-root>. Preserve all other settings in FIXED_WALL_3PDE_20260911.md: same three PDEs, paired seed 20260918, 600 seconds per method (six cells), batch400, Adam constant lr1e-4, same losses/constraints/evaluation points. Existing baseline artifacts remain immutable. Run tests and all six two-step full-width smoke cells before formal training. No additional seeds or retries.
