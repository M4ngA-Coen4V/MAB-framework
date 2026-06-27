# Thesis Results Summary

Seeds: 101, 202
Evaluation episodes per seed: 2
Steps per episode: 20

## RQ1 Dynamic Behavior
Average switch interval: 1.00
Std dev of switch interval: 0.00
Resource wealth at switch (mean): 68.94
Surviving agents at switch (mean): 30.00
Switch point array: [1, 1, 1, 1]

## RQ2 Performance vs Baselines
- PPO: rewards=[1252.7876906619174, 1114.2522783203178]
  extracted=59.18, degradation=28.98, oppCost=30.34
- Pigouvian: rewards=[1139.9420988511638, 1347.475487592678]
  extracted=62.19, degradation=31.63, oppCost=24.60
- Free Market: rewards=[1143.5014421047686, 1344.4488817990427]
  extracted=62.20, degradation=31.42, oppCost=24.85
- Wealth Multiplier: rewards=[1343.8906192810628, 1357.0596364170583]
  extracted=67.52, degradation=36.24, oppCost=22.52
- Progressive: rewards=[1136.4974956308647, 1316.4801168263407]
  extracted=61.32, degradation=27.32, oppCost=27.86
Theoretical episode ceiling: 1988.00

## RQ3 Learning Stability
All PPO seeds above 49000: False
Checkpoint reward values: [1323.7528879911256, 1323.7528879911256, 1323.7528879911256]
Checkpoint critic loss values: [0.12973900139331818, 0.12973900139331818, 0.12973900139331818]
Checkpoint explained variance values: [0.0129164457321167, 0.0129164457321167, 0.0129164457321167]
Checkpoint entropy values: [1.385273238023122, 1.385273238023122, 1.385273238023122]
Checkpoint KL values: [4.460278434000505e-06, 4.460278434000505e-06, 4.460278434000505e-06]

## Mechanism
PPO elite survivor correlation (mean): 0.5321
Pigouvian elite survivor correlation (mean): 0.5734
Wealth Multiplier elite survivor correlation (mean): 0.5912

## New Survivorship Bias Data (PPO only)
Elite r at step 10: 0.0132
Starved r at step 10: nan

## New Arm Selection Data (PPO, Free Market, Progressive)
PPO Early Tier 1: 18.58
PPO Late Tier 1: nan
PPO Late Tier 2: nan
PPO Late Tier 3: nan

Free Market Early Tier 1: 22.58
Free Market Late Tier 1: nan
Free Market Late Tier 3: nan

Progressive Early Tier 1: 20.33
Progressive Late Tier 1: nan
Progressive Late Tier 2: nan
Progressive Late Tier 3: nan