# Real-robot structured dataset

`learm_demonstrations_structured.zip` contains the complete structured logs for the 24 physical LeArm demonstrations used in this project: 16 success episodes and 8 failure episodes.

The raw camera videos are intentionally omitted from this repository because the full media archive is approximately 3.27 GB and is not required to retrain the current policy. The structured archive retains the episode metadata, reconstructed tool trajectory, synchronized visual measurements, and task-phase fields used by the training pipeline.

The trainer consumes this complete archive directly and performs its own success/sample filtering.
