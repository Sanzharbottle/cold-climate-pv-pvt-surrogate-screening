# Cold-Climate PV/PVT Surrogate Screening

This repository contains the code and model outputs supporting the manuscript:

> Serikbolov, S. *Surrogate Multi-Objective Screening of a Conceptual Cold-Climate PV System with Thermal-Protection Context: Yield, Water Use, and Cost Trade-offs.*

## What this repository contains

- `run_manuscript_results.py` — main model code
- `requirements.txt` — required Python packages
- `results/` — outputs used in the manuscript

## Important note

This is a conceptual surrogate screening model.

It is not:

- a field-validated PV/PVT model;
- a complete life-cycle assessment;
- a calibrated battery-life forecast;
- a bankable engineering design.

## How to run

Install Python 3.12, then run:

```bash
pip install -r requirements.txt
python run_manuscript_results.py
```

The results will be written into the `results/` folder.

## Main modeled compromise point

- Summer tilt: 28°
- Winter tilt: 62°
- Cleaning interval: 18 days
- Thermal-buffer setpoint: 40°C

At this point, the model gives:

- 20-year electrical yield: 276.4972 GWh
- Physical cleaning-water use: 10,670 m³/year
- Thermal-protection proxy: 82.4%

## Baseline note

The 242.1 GWh unheated-baseline yield is an external planning-level scenario input used for the LCOE comparison. It is not calculated by the yield surrogate.

## Citation

If you use this code or results, please cite:

Serikbolov, S. (2026). *Surrogate Multi-Objective Screening of a Conceptual Cold-Climate PV System with Thermal-Protection Context: Yield, Water Use, and Cost Trade-offs.*

## License

Apache License 2.0.
