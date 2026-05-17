# Wave-Energy Converter Sensor Dashboard

Physics-based simulation of float sensors with anomaly detection, structural-health monitoring, and fatigue-life estimation — a fleet-engineering prototype.

**🌊 Live demo (no install needed):** [wec-sensor-dashboard.onrender.com](https://wec-sensor-dashboard.onrender.com/)

<!-- VIDEO_PLACEHOLDER: paste the GitHub user-attachment URL here. See "Adding the demo video" below for how. -->

---

## What's in it

Six tabs walk through the sensor stack on a free-drifting wave-energy converter.

- **Overview** — float architecture and the six sensor categories on board
- **Motion** — accelerometer and gyroscope
- **Strain & Fatigue** — strain gauge at the welded neck joint, with a 20-year fatigue-life verdict across sea states
- **Resonance** — natural-frequency identification from the strain time series, plus a forcing sweep that shows when hard limits (yield / UTS) start to break
- **Pressure** — five channels (three external hull sensors, one internal power-take-off chamber, one dry-cabin leak detector), with wave-to-wire power output
- **ADCP** — acoustic current profiler with GPS-fusion correction

Every tab has interactive sliders so you can play with wave dynamics, engineering parameters, and design thresholds — and watch the figures, formulas, tables, and verdicts update in real time.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8050

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

You are free to use, modify, and redistribute it for **personal, research, or educational purposes**, including hobby projects and study. **Commercial use is not permitted.** See the [LICENSE](LICENSE) file for full terms.

If you build on this work, please retain attribution to the author.

## Author

Youran Li, 2026
