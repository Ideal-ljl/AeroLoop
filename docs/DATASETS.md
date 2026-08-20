# Dataset adapters

Dataset adapters only translate files into `EpisodeSpec`; they do not start or
wrap a simulator. This keeps OpenFly, TravelUAV, and future formats independent
from simulator integration packages.

## OpenFly

The loader reads the test-list JSON (`path`, `vla_caption` or
`gpt_instruction`) and each trajectory's `pose_bbox_updated.json`. Relocated
absolute paths can be repaired with `path_rewrites`.

## TravelUAV

The loader reads a split such as `seen_valset.json` or `unseen_valset.json`,
then resolves every referenced `merged_data.json`. Official splits contain one
row per frame; `deduplicate: true` produces one evaluation episode per
trajectory. Set it to false for frame-level samples.

## Inspection

```bash
aeroloop inspect-dataset --config configs/datasets/openfly.yaml
aeroloop visualize-dataset --config configs/datasets/traveluav.yaml \
  --output eval_results/dataset.html --limit 100
```

The self-contained HTML preview shows instructions, XY paths, start/target
metadata, scene names, and path lengths without requiring a simulator.
