# Validation performed while assembling this package

The following checks were actually run after the repository rename/refactor:

- all retained Python source files passed syntax compilation;
- `real_robot_dataset/learm_demonstrations_structured.zip` contains the complete structured 24-episode dataset (16 success, 8 failure);
- the canonical trainer was run from the renamed repository layout and completed successfully;
- that run selected the 16 successful episodes, constructed 8,252 valid pre-blue-loss samples, and reproduced a grouped-CV mean vector RMSE of approximately 3.86694 mm per 100 ms;
- the freshly trained checkpoint satisfied the same 5-feature / 3-action / 100 ms runtime policy contract;
- the bundled checkpoint was then restored to the previously simulation-validated reference checkpoint with SHA256 `5eddf8684168f66ae5fa4b5e5515c45e795267b0adbb2fdb177a4665997bd3c2`.

The complete MuJoCo physics/GUI smoke test was **not** executed in the assembly environment because MuJoCo is not installed there. Run the following on the target development machine after installing `requirements.txt`:

```bash
python evaluation/smoke_test_panda_simulation.py
python run/run_learned_pre_insertion_policy.py --preset baseline
python run/run_panda_hybrid_ring_pick.py --preset baseline
```
