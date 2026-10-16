# Data and policy provenance

The repository keeps the real-data-to-policy chain explicit:

```text
real_robot_dataset/learm_demonstrations_structured.zip
        ↓
behavior_cloning_training/train_blue_relative_pre_insertion_policy.py
        ↓
trained_policy/blue_relative_pre_insertion_policy.npz
        ↓
run/run_panda_hybrid_ring_pick.py
```

## Bundled artifacts

- Structured dataset SHA256: `f78de188fe4a1597b67ad5f26172ae2781b0ff6c4c5a8baf090fd53d6d90b495`
- Bundled validated policy SHA256: `5eddf8684168f66ae5fa4b5e5515c45e795267b0adbb2fdb177a4665997bd3c2`
- Canonical trainer SHA256: `fa7c8aa80f7a15e2e0dd15becb710ec1b23106045d0206aeaffba96e98e1d4df`

The structured dataset contains all 24 collected LeArm episodes: 16 success and 8 failure. The trainer selects successful episodes and valid pre-insertion windows internally.

The canonical training output path and the default simulation model path are intentionally the same: `trained_policy/blue_relative_pre_insertion_policy.npz`. Running the trainer therefore replaces the bundled reference policy with the newly trained policy.

Fixed random seeds make the data construction and evaluation repeatable, but exact checkpoint bytes are not claimed to be identical across different numerical/PyTorch environments.
