# Current simulations only

Use MTSManager's `MTS_Management.config.Config` for settings, names and hashes.
Keep the dependency one way: MTSManager must not import MTSPlot.
Do not import SimulationScripts or add legacy readers or aliases.
Raise errors for inconsistent identities, malformed tables and invalid timings.
Prefer small reusable functions over plotting classes and wrappers.
Use small individual vector PDFs with readable text and markers.
Do not perform automated visual checks: do not render PDFs to images, inspect
plot images, make preview/contact sheets, or iterate based on their appearance.
Make your best attempt at the plotting code, run it, and fix runtime errors.
The user will review the plots and explicitly request any visual changes.
Keep data retrieval explicit; plotting must never submit or modify simulations.
Preserve the distinction between partial data, strain-limit stops and target completion.
Run `python -m unittest discover -s tests -v` with `MPLBACKEND=Agg`.
