# RelayGrid

RelayGrid is a deterministic Python toolkit for distribution-grid operations.
It covers feeder topology, protection, voltage assessment, DER islanding,
switching safety, outage accounting, crew logistics, asset thermal limits,
telemetry quality, storage dispatch, and restoration sequencing.

The package is self-contained and uses only the Python standard library.

## Install and run

```bash
python -m pip install -e ".[test]"
relaygrid demo
python -m pytest -q
```

The demo builds a four-bus feeder with two open ties, reports its energized
state, safely closes the sectionalizing branch, and emits a JSON summary.

## Public API

The top-level package exports the feeder data model and `GridSimulator`:

```python
from relaygrid import Branch, Bus, FeederSnapshot, GridSimulator, SwitchState
```

Domain modules remain independently importable so applications can combine
topology, protection, telemetry, restoration, reliability, planning, and field
operations without depending on a web framework or database.

### Dynamic line capacity

`relaygrid.conductors.dynamic_rating` converts conductor parameters, ambient
temperature, wind speed, solar heating, and current loading into a time-limited
available capacity. Missing, invalid, or expired weather evidence falls back to
the conductor's static ampacity. Batch results identify each derating cause;
calculations are returned as separate capacity records and never overwrite the
catalog's static equipment rating.

## License

RelayGrid is released under the MIT License.
