# Method summary

1. Real LeArm demonstrations are collected with eye-in-hand red-ring/blue-tip perception and a firmware-synchronized host-side shadow kinematic state.
2. Structured trajectories store task-space tool pose plus compact visual observations.
3. A small behavior-cloning MLP learns only the visually coupled pre-insertion phase from successful real demonstrations.
4. The learned output is a 100 ms tool-local translation `[left, up, forward]`, independent of LeArm/Panda joint coordinates.
5. In Panda simulation the same compact blue-relative visual observation drives the MLP.
6. Sustained loss of the blue terminal marker is used as the insertion event; after this event control switches to a deterministic upward-lift + mild pitch primitive.
