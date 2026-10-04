| # | Clip | Condition | Seed | Verdict (primary detector) |
|---|---|---|---|---|
| 1 | MAP-Elites three-phase controller — nominal | nominal | 10000 | success |
| 2 | MTR-PPO, CTBR actions — nominal | nominal | 10000 | success |
| 3 | MTR-PPO, motor commands — nominal | nominal | 10000 | success |
| 4 | MTR-PPO, motor commands — stress | ppo_track_stress | 10000 | success |
| 5 | MTR-PPO, CTBR actions — stress | ppo_track_stress | 10001 | FAIL: never settled upright and still |
| 6 | MAP-Elites three-phase controller — stress | ppo_track_stress | 10000 | FAIL: never settled upright and still |
