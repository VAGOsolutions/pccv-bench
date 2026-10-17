# E1 — tiers (easy / medium / hard), aligned to the frozen operating point

- git revision: `af0ee770569a0cc5e5a2b3ffe55175cb15fc1abe`
- tier git revisions: {'easy': 'f77df32c5a09408d83a09a8de13e4a9f83b4e69b', 'medium': 'f77df32c5a09408d83a09a8de13e4a9f83b4e69b', 'hard': 'f77df32c5a09408d83a09a8de13e4a9f83b4e69b'}

## Tier-expectation check (frozen verifier `+all+twoway+offset_aware`)

- pooled detection (easy, medium, hard): [0.9563, 0.9635, 0.9604] -- non-increasing: **False**
- clean flag rate (easy, medium, hard): [0.1875, 0.3542, 0.4708] -- non-decreasing: **True**
- both hold: **False**

## Full ladder

| arm | easy detect | medium detect | hard detect | easy clean | medium clean | hard clean |
|---|---|---|---|---|---|---|
| `flag_none` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `flag_all` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `baseline_envelope` | 0.9771 | 0.9771 | 0.9771 | 0.3792 | 0.4792 | 0.5333 |
| `baseline_symmetric` | 0.9938 | 0.9885 | 0.9833 | 0.4625 | 0.5375 | 0.5667 |
| `baseline_coherent` | 0.9927 | 0.9844 | 0.9802 | 0.4583 | 0.5250 | 0.5625 |
| `baseline_pairwise` | 0.9885 | 0.9854 | 0.9823 | 0.4417 | 0.5292 | 0.5625 |
| `geometry` | 0.7312 | 0.7115 | 0.6625 | 0.1292 | 0.2125 | 0.2125 |
| `+offset` | 0.7240 | 0.6948 | 0.6510 | 0.1208 | 0.1792 | 0.1958 |
| `+temporal` | 0.7292 | 0.7115 | 0.6594 | 0.1250 | 0.2083 | 0.2042 |
| `+offset+temporal` | 0.7219 | 0.6948 | 0.6500 | 0.1167 | 0.1750 | 0.1917 |
| `geometry+twoway` | 0.9740 | 0.9781 | 0.9740 | 0.2000 | 0.3875 | 0.4875 |
| `+offset+twoway` | 0.9708 | 0.9719 | 0.9688 | 0.1958 | 0.3583 | 0.4750 |
| `+temporal+twoway` | 0.9740 | 0.9781 | 0.9740 | 0.1958 | 0.3875 | 0.4833 |
| `+all+twoway` | 0.9708 | 0.9719 | 0.9688 | 0.1917 | 0.3583 | 0.4750 |
| `+all+twoway+offset_aware` | 0.9563 | 0.9635 | 0.9604 | 0.1875 | 0.3542 | 0.4708 |
