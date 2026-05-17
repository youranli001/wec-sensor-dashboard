# Wave-Energy Converter Sensor Dashboard

Physics-based simulation of float sensors with anomaly detection, structural-health monitoring, and fatigue-life estimation — a fleet-engineering prototype.

**Live demo:** _add your Render URL here once deployed_

---

## What's in it

Six tabs walk through the sensor stack on a free-drifting wave-energy converter.

- **Overview** — float architecture and the six sensor categories on board
- **Motion** — accelerometer and gyroscope
- **Strain & Fatigue** — strain gauge at the welded neck joint
- **Resonance** — natural-frequency identification from the strain time series
- **Pressure** — five channels (three external hull sensors, one internal power-take-off chamber, one dry-cabin leak detector)
- **ADCP** — acoustic current profiler with GPS-fusion correction

Every tab has interactive sliders so you can play with wave dynamics, engineering parameters, and design thresholds — and watch the figures, formulas, tables, and verdicts update in real time.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8050
