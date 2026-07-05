# Paper Report Data

This directory is generated from the legacy MESA-Web job by:

```powershell
.venv\Scripts\python.exe scripts\build_paper_report_data.py
```

The numbered folders follow all of Chapter 3 on PDF pages 18-42 of the provided
research-paper reference. `METHOD_TRACEABILITY.csv` maps each method to its
evidence, and `MANIFEST.csv` records file sizes and SHA-256 checksums.

Important interpretation notes:

- These eight snapshots are from one approximately solar-mass MESA track. They
  are implementation evidence, not the proposed 50,000-model atlas.
- The PINN is not trained; no synthetic training result is presented as evidence.
- Hydrostatic failures are retained as findings rather than silently removed.
- Radiation uncertainty uses a recorded 1% input-error assumption. Seeded
  bootstrap summaries record completion and successful-fit rates per profile.
