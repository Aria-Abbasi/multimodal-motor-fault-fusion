# NLN-EMP Vibration Sensor Selection

## Frozen Selection

Use **vibration channel 2** for all NLN-EMP experiments.

- Physical location: electric-motor driven-end bearing
- Direction: vertical
- Sampling rate: 20 kHz
- Unit: g

## Dataset Channel Map

The dataset README supplied with the official 4TU archive defines:

1. Electric-motor non-driven-end bearing, horizontal
2. Electric-motor driven-end bearing, vertical
3. Electric-motor driven-end bearing, axial
4. Pump driven-end bearing, horizontal
5. Pump non-driven-end bearing, vertical

Electric channels 1-3 are the three phase currents. Electric channels 4-6 are
the three phase voltages.

## Rationale

Channel 2 is closest to the motor driven-end bearing and motor-pump coupling
load path. Its vertical orientation is also aligned with the primary bearing
load direction. This makes it a defensible single-sensor choice for the
paper's mixed motor, bearing, shaft, alignment, and unbalance fault set.

The channel was selected from physical sensor documentation before examining
corrected model test results. It must remain fixed across folds, seeds,
baselines, and ablations.

## Verification

Verified on June 13, 2026 using:

- `data/raw/nln_emp/README.txt` from the official 4TU archive;
- extracted paths under `data/raw/nln_emp/Dataset/Vibration`;
- representative `-ch1.csv` through `-ch5.csv` recording files.
