# Nightmare (WIP)

> 🚧 **Work in progress.** Nightmare is under active development and not yet
> functional. The design, configs and interfaces below may change.

**Failure-aware world models: imagine failure states, then avoid them.**

Nightmare is a fork of [DreamerV3][dreamerv3] for control problems where
certain states are *failure states*, specific combinations of numerical values that
must be avoided. The agent learns a world model of the system from a
simulator, imagines possible futures, predicts which of them end in a failure
state, and trains a policy that stays away from them.

## Motivation

DreamerV3 learns a world model and trains an actor-critic policy inside
imagined trajectories. It handles vector observations natively and already
predicts episode termination. Nightmare builds on this for settings where:

- observations are mainly **numerical state vectors** at every step, not
  video;
- some states are **terminal failure states**, defined by combinations of
  values;
- failure states are **costly**, so the policy should avoid them, not just trade them
  off against reward;
- a **simulator** is available for online interaction.

Rendering numbers into images to use video world models (such as Dreamer 4)
adds no information and wastes compute. Nightmare models the numbers
directly.

## Method

Nightmare keeps the DreamerV3 core (RSSM world model, symlog predictions,
imagination-based actor-critic, fixed hyperparameters) and adds:

1. **Failure head.** A dedicated head predicts the probability that a state is
   a failure state. It extends DreamerV3's continue head, but is trained
   separately on failure labels from the simulator.
2. **Rule-based failure check.** When the failure condition is known exactly, it
   is also applied to the decoded numerical predictions. A state is flagged
   if either the rule or the learned head flags it.
3. **Failure-terminated imagination.** Imagined rollouts end at predicted failure
   states and receive a large penalty, so value estimates propagate risk
   backwards and the policy steers away early.
4. **Ensemble uncertainty.** An ensemble of dynamics/failure heads; a state is
   treated as risky if any member predicts a failure state (pessimistic, minimax-style
   imagination). High disagreement is also penalized, because it marks
   futures the data doesn't cover.
5. **Failure-focused replay.** Sequences that contain or approach failure states
   are oversampled, so the world model learns rare failures. Optional
   simulator resets into near-failure states increase coverage.
6. **Constraint mode (optional).** Failure states are treated as a cost constraint
   (Lagrangian, cost ≤ limit) instead of only a negative reward, for settings
   where failure states must essentially never happen.
7. **Deployment safety filter.** A hard, rule-based check outside the learned
   model that vetoes actions predicted to lead into failure states. Learned
   models cannot guarantee avoidance, so this is the backstop.

Optional: a terminal image (if the system produces one) is used only as an
outcome label or reward signal, not as a per-step observation.

## Instructions

The code is based on DreamerV3 and requires Python 3.11+ and [JAX][jax].

```sh
pip install -U -r requirements.txt
```

Training (provisional flags):

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/nightmare/{timestamp} \
  --configs nightmare \
  --task <your_simulator_task> \
  --run.train_ratio 32
```

Your environment should provide, per step:

- `vector`: the numerical observation (any scale; symlog handles it);
- `reward`: scalar reward;
- `is_terminal`: episode end;
- `is_failure`: whether the state is a failure state.

View results:

```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

## Evaluation

Report, alongside return:

- **failure rate** per episode, in the simulator and on held-out conditions;
- **failure-head calibration** (precision/recall, especially recall on failure states);
- **steps-before-failure margin**: how early the policy steers away;
- comparison against a **PPO baseline** and plain DreamerV3.

## Roadmap

- [ ] `is_failure` observation key and failure head
- [ ] Failure-terminated imagination with penalty
- [ ] Ensemble heads and pessimistic risk aggregation
- [ ] Failure-focused replay prioritization
- [ ] Lagrangian constraint mode
- [ ] Deployment safety filter
- [ ] Domain randomization for sim-to-real transfer

## Credits

Nightmare is a fork of [DreamerV3][dreamerv3] by Danijar Hafner et al. If you
use this code, please cite the original work:

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

This project is unrelated to Google or DeepMind.

[dreamerv3]: https://github.com/danijar/dreamerv3
[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
