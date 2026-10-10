# AAPA Phase 2A Repaired Simulation Smoke Test

These files are **a reconstructed simulator**, not the original AAPA paper implementation or faithful reproduction results. They leave `src/core/simulator.py` untouched.

Copy `src/core/simulator_repaired.py` and `examples/phase2_repaired_smoke.py` into the matching locations in your repository.

From the root of your Python virtual environment:

```powershell
python examples/phase2_repaired_smoke.py
```

Results are written to `results/phase2_repaired_smoke/summary.json` and per-policy timelines.

## Important experimental distinctions

- Unlike the original generator, the replay generator uses *exact rounded request counts per minute*, with randomized uniform arrival times and exponential processing durations precomputed once and shared across policies.
- Each replica is a SimPy worker able to process one request at a time; additional replicas add concurrency. Starting replicas are not immediately ready: 30-second startup.
- All strategies use the same load units, requests/second, and the same calibrated processing demand (0.1 CPU-second per request / mean request service duration).
- Comparison policies are **normalized experimental versions** of HPA and rolling-trend predictive scaling, not exact repository algorithms.
- Unfinished requests at the end count as violations in the explicitly named inclusive metric; P99 summarizes completed requests only.
- The original AAPA constructor can load your exported LightGBM Booster, but connecting the full trained classifier requires temporally valid feature windows and is not part of this first smoke test. Synthetic classifier features are not valid evidence.

## Caveats / next validation requirements

1. Validate timestamp causality, load observation time, and forecasting horizon. A rolling 60-second rate sampled every 60 seconds is used here.
2. The replica scale-down procedure retires idle replicas only. Busy workers are allowed to finish; the desired count may not be attained immediately. Track this from the timeline.
3. Pod-minutes integrate allocated replicas over time, including startup. They are not billed resource metrics.
4. This is one simulated concurrent request per pod; actual Kubernetes/serverless concurrency and CPU usage may differ. Derive service model and startup/config values from the paper before evaluating Azure traces.
5. Run multiple deterministic seeds and include experiment settings, failure accounting, and CI tests in the final baseline.
6. The original simulator's `pod_warmer` made pods warm within ~1 second despite the declared 2-second startup; that semantics is intentionally replaced by a real 30-second startup.
