# Ocean-3 Sensor Dashboard

A Plotly Dash app simulating fleet-monitoring for a wave energy converter (WEC). Each tab covers one sensor category — ground truth vs measurement, plus the analysis pipeline that turns raw data into structural health signals.

## Tabs

0. **Overview** — schematic + Ocean-3 design parameters
1. **Motion** — accelerometer + gyro with bias drift
2. **Strain + Dynamics** — V4 resonance demo + FFT-based system identification (the showpiece)
3. **Pressure** — 5 channels (hull at 3 depths, internal PTO, differential, dry cabin)
4. **ADCP** — depth-profiled currents + GPS-fused velocity correction
5. **Temperature** — 5 channels + bearing seal fault detection via rolling-mean drift
6. **GPS** — trajectory, geofencing, hardware-freeze anomaly detection

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8050

## Deploy to Render

1. Push to GitHub
2. New web service on Render, connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:server`
5. Plan: free is fine but 15-min idle spindown applies

## Design notes

- **All data pre-computed at module load** — Render free tier (0.1 CPU) won't recompute per-callback. One pass at startup, cached in `DATA` dict.
- **Single callback for tab content** — explicit pattern, no hidden state
- **Restrained styling** — no emoji, minimal color, monospace for numerical parameters

## Performance

Tab switch time is dominated by Plotly rendering, not data computation. On free-tier Render, expect 1-2 s per tab after first load.

## Acknowledgments

Built as a portfolio piece for fleet sim engineer applications. Sensor models, fault scenarios, and physics framework based on standard WEC analysis literature (BS 7608, DNV-GL, NEMOH).
