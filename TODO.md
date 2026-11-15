# Nightmare TODO

Ordered so that each step can be trained and checked before the next one
starts. File references are to the DreamerV3 code as forked.

## 0. Groundwork

- [ ] Write the simulator env in `embodied/envs/<name>.py`, following
      `embodied/envs/dummy.py` (`obs_space`, `act_space`, `step`).
  - [ ] Return `vector`, `reward`, `is_first`, `is_last`, `is_terminal`,
        `is_failure` on every step.
  - [ ] On the step where the episode ends, return the real final (failure)
        observation. Reset only on the next step.
  - [ ] Declare finite action bounds.
  - [ ] Rescale or offset every value a failure rule depends on, so the
        threshold sits near zero at order 1 (symlog has little resolution
        for large values).
- [ ] Register the env in the `ctor` dict in `dreamerv3/main.py` (`make_env`).
      The task name needs an underscore, e.g. `<name>_default`.
- [ ] Add a `nightmare` block to `dreamerv3/configs.yaml` (the README already
      uses `--configs nightmare`): task, model size, `run.train_ratio`,
      `env.<name>` arguments.
- [ ] Run plain DreamerV3 on the simulator with `--configs nightmare debug`
      and then at full size. This is the baseline for every later comparison.
- [ ] Log failure rate per episode from the env (`log/` keys or `logfn` in
      `embodied/run/train.py`).

## 1. Failure head

- [ ] Add `is_failure` to `exclude` in `Agent.__init__`
      (`dreamerv3/agent.py`), so it is neither encoded nor reconstructed.
- [ ] Add `self.fail = MLPHead(binary, **config.failhead, name='fail')` next
      to `self.con`, and add it to `self.modules`.
- [ ] Add `losses['fail']` in `Agent.loss`, next to `losses['con']`. Use the
      raw label; do not apply the `contdisc` scaling.
- [ ] Add `failhead` and `loss_scales.fail` to `configs.yaml`.
- [ ] Add precision, recall and calibration of the failure head to
      `Agent.report`.
- [ ] Decide how to handle class imbalance (loss weighting or replay, see 5).

## 2. Rule-based failure check

- [ ] Define the rule interface: a function from decoded observations (raw
      units) to a boolean, supplied per task.
- [ ] In `Agent.loss`, run `self.dec` on `imgfeat` and apply `nn.symexp` to
      the predictions before applying the rule.
- [ ] Combine: flagged if rule OR head.
- [ ] Measure the extra compute of decoding in imagination; make it a config
      switch.

## 3. Failure-terminated imagination

- [ ] In `Agent.loss`, before the `imag_loss` call: multiply `con` by
      `(1 - p_fail)` and subtract `penalty * p_fail` from `rew`.
- [ ] Add `fail_penalty` to `configs.yaml`.
- [ ] Apply the same treatment in the replay value loss (`repl_loss`), using
      the real `is_failure` labels.
- [ ] Check how the penalty interacts with return normalization (`retnorm`
      uses the 5-95 percentile range).
- [ ] Metric: steps-before-failure margin.

## 4. Ensemble uncertainty

- [ ] Failure head ensemble: N heads with different initializations;
      aggregate with max (pessimistic).
- [ ] Dynamics ensemble: N `_prior` networks in `dreamerv3/rssm.py`; decide
      which member `imagine` samples from.
- [ ] Disagreement measure and penalty term in the imagined reward.
- [ ] Config: ensemble size, disagreement scale.

## 5. Failure-focused replay

- [ ] Choose the mechanism: the existing `Prioritized` selector
      (`embodied/core/selectors.py`) fed with a failure-based priority, or a
      dedicated selector that indexes sequences containing or approaching
      failure.
- [ ] If `Prioritized`: emit `outs['replay']['priority']` with shape (B, T)
      from `Agent.train` (stub at the commented lines), and set
      `replay.fracs` in the config.
- [ ] Correct for the sampling bias in the failure head and world model loss
      (importance weights), or accept and document the calibration shift.
- [ ] Optional: simulator resets into near-failure states.

## 6. Constraint mode (Lagrangian)

- [ ] Cost critic: second value head trained on imagined failure cost.
- [ ] Lagrange multiplier as an `nj.Variable`, updated from
      (estimated cost - limit).
- [ ] Actor advantage = reward advantage - multiplier * cost advantage.
- [ ] Config switch between penalty mode and constraint mode.

## 7. Deployment safety filter

- [ ] Deterministic policy mode: `Agent.policy` ignores `mode` and always
      samples; add a mean/argmax path for evaluation and deployment.
- [ ] Rule-based filter around `agent.policy` on the NumPy side, with a
      fallback action when the proposed action is vetoed.
- [ ] Log veto rate.

## 8. Evaluation

- [ ] PPO baseline on the same simulator.
- [ ] Held-out simulator conditions for failure rate.
- [ ] Domain randomization for sim-to-real.
- [ ] Ablations: each of steps 1-6 switched off individually.

## Housekeeping

- [x] `Agent.report` dropped its loss metrics (`mets.update(mets)`).
- [x] `run.from_checkpoint_regex` was missing from the config.
- [ ] `make_env` points `dm` at `embodied.envs.from_dmenv`; the file is
      `from_dm.py` (only matters for the `dm` suite).
- [ ] `np.ranodm` typo in `embodied/core/streams.py` (`Mixer`, unused).
- [ ] Update banner, package name and Dockerfile for Nightmare.
