# Current simulations only

Write README files for first-time users with no access to chat history. Introduce
the project's purpose, prerequisites and terminology before giving examples.
Do not recount implementation decisions, completed requests, migrations or past
conversations. Keep the README concise and task-oriented; present plotting scripts
as examples and introduce any specialized campaign before referring to it.

Only add high-value tests that catch plausible scientific or data-handling errors,
such as incorrect timing statistics, accidental mixing of runs, or mathematical
derivative errors. Do not add tests for trivial helpers, filenames, plot counts,
cosmetics or straightforward forwarding merely to have tests. Small changes often
need no new test. Remove redundant existing tests and prefer regressions for bugs
actually encountered. Do not automate plot appearance checks.

Use MTSManager's `MTS_Management.config.Config` for settings, names and hashes.
Keep the dependency one way: MTSManager must not import MTSPlot.
Do not import SimulationScripts or add legacy readers or aliases.
Raise errors for inconsistent identities, malformed tables and invalid timings.
Prefer small reusable functions over plotting classes and wrappers.
Use small individual vector PDFs with readable text and markers.
Use standard Matplotlib legends with their box/frame enabled. Default to
loc="best" and allow the number of legend columns to be specified. Do not build
custom legends or move them outside the axes automatically. Expose legend
placement and axis limits for the user to tune manually; do not iterate on them
using visual checks.
Do not perform automated visual checks: do not render PDFs to images, inspect
plot images, make preview/contact sheets, or iterate based on their appearance.
Make your best attempt at the plotting code, run it, and fix runtime errors.
The user will review the plots and explicitly request any visual changes.
Keep data retrieval explicit; plotting must never submit or modify simulations.
Preserve the distinction between partial data, strain-limit stops and target completion.
Run `python -m unittest discover -s tests -v` with `MPLBACKEND=Agg`.
