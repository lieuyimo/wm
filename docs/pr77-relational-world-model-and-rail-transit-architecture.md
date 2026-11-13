# Deprecated research interpretation: railway is NOT the requested transport mode

User clarified on November 14, 2026: the planned extension is **automated laboratory sample conveyor / track carrier transfer**, NOT railways, passenger trains or railway signal control. The previous Flatland / RSTGCN / train-scheduling migration described here was based on a mistaken interpretation and must not be used for this project.

**Canonical revised architecture:** [PR77 laboratory sample conveyor and relational world-model architecture](pr77-relational-world-model-and-sample-conveyor-architecture.md).

The original PR61 relation world-model backbone stays; `world_model_core` is now a single `lab_sample_transport` public graph schema capable of co-representing AMR transport and conveyor-carrier sections. Schema representation is not already-trained prediction or validated conveyor simulator dynamics.
