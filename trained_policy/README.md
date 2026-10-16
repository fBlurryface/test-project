# Trained policy

`blue_relative_pre_insertion_policy.npz` is the policy consumed by the Panda hybrid runtime.

It is a NumPy-exported MLP with the observation normalization, action de-normalization, feature names, action names, and 100 ms policy horizon stored in the same artifact.

To replace it with a freshly trained policy:

```bash
python behavior_cloning_training/train_blue_relative_pre_insertion_policy.py
python evaluation/check_policy_interface.py
```

Then run the complete hybrid task with:

```bash
python run/run_panda_hybrid_ring_pick.py --preset baseline
```
