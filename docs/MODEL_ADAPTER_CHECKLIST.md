# Model adapter checklist

Before a model is admitted to a benchmark run, its adapter must document and
test all of the following:

1. Required named camera views, image color order, resolution, and per-view history policy.
2. Absolute versus start-relative state and the unit of every state channel.
3. Native output layout, units, coordinate frame, and chunk length.
4. Conversion to `[dx_body,dy_body,dz_body,d_yaw,stop_probability]`.
5. Stop semantics and threshold; a parse failure must be reported as an error,
   not silently converted to stop.
6. Episode reset behavior, caches, random seeds, and deterministic settings.
7. A fixture with at least one known input/output pair.
8. A service health check and model/checkpoint identity in response metadata.
9. Peak GPU memory and inference latency after warmup.
10. Both `execution_horizon=1` and native-chunk smoke runs.
11. Behavior when a required view or auxiliary state field is unavailable.
