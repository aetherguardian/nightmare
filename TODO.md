# Nightmare TODO

**Before env**: can be built and tested now, with the dummy env.
**After env**: needs the real simulator env (its data, failure rules or
behaviour).

## 0. Groundwork

- [ ] **After env** Write the simulator env in `embodied/envs/<name>.py`,
      following `embodied/envs/dummy.py` (`obs_space`, `act_space`, `step`).
  - [ ] Return `vector`, `reward`, `is_first`, `is_last`, `is_terminal`,
        `is_failure` and `log/progress` on every step.
  - [ ] Observations are vectors only: float32 arrays with 1 or 2 dims. No
        3-dim `uint8` arrays (those are routed to the CNN image path).
  - [ ] Use float for every continuous or count-like feature. Integer
        features are one-hot encoded and decoded as categories, so use int
        only for true categories with few classes.
  - [ ] Rescale or offset every value a failure rule depends on, so the
        threshold sits near zero at order 1 (symlog has little resolution
        for large values).
  - [ ] Declare finite action bounds.
  - [ ] Episode ends: a failure sets `is_failure`, `is_terminal` and
        `is_last`; reaching the goal sets `is_terminal` and `is_last`; a
        timeout sets only `is_last`. Return the real final observation on
        that step and reset only on the next one.
  - [ ] Reward: define a progress measure Φ(state), 0 at the start and 1 at
        the goal. Reward the change per step,
        `shaping_scale * (Φ(s_t) - Φ(s_t-1))`, plus `goal_bonus` when the goal
        is reached. Never reward Φ itself per step (the agent would learn to
        sit near the goal instead of reaching it). No failure term in the
        reward; the penalty lives in the agent (section 3).
  - [ ] `log/progress` is Φ: logged per episode, never seen by the agent,
        used to rank candidates (section 6).
- [ ] **Before env** Add `is_failure` to `embodied/envs/dummy.py` (always
      False) so the debug smoke test runs.
- [ ] **Before env** Log failure rate per episode (`logfn` in
      `embodied/run/train.py`).
- [ ] **After env** Register the env in the `ctor` dict in `dreamerv3/main.py`
      (`make_env`). The task name needs an underscore, e.g. `<name>_default`.
- [ ] **After env** Add a `nightmare` block to `dreamerv3/configs.yaml` (the
      README uses `--configs nightmare`): task, model size,
      `run.train_ratio`, `env.<name>` arguments.
- [ ] **After env** Pick a small model size. The defaults (size200m) are
      sized for images; the vector preset `dmc_proprio` uses `size1m`. Start
      from `size1m` or `size12m` and sweep. The image settings (`depth`,
      `mults`, `kernel`, `bspace`) have no effect on vector inputs.
- [ ] **After env** Run plain DreamerV3 on the simulator with
      `--configs nightmare debug` and then at full size. This is the
      baseline for every later comparison.

## 1. Failure head

- [x] Add `is_failure` to `exclude` in `Agent.__init__`, so it is neither
      encoded nor reconstructed.
- [x] Add `self.fail` (binary `MLPHead`) next to `self.con`, and add it to
      `self.modules`.
- [x] Add `losses['fail']` in `Agent.loss` on the raw label (no `contdisc`
      scaling), fed with `repfeat` without `sg` so the failure loss also
      trains the latent.
- [x] Add `failhead` and `loss_scales.fail` to `configs.yaml`.
- [x] Precision, recall and calibration metrics in `Agent.loss` (also shown
      in reports).
- [ ] **After env** Decide how to handle class imbalance (loss weighting or
      replay, section 5); depends on the real failure rate.

### Keep failure-relevant information in the latent

- [ ] **Before env** Give observation keys their own reconstruction weight.
      Today every decoded key gets `loss_scales.rec` (`self.scales` in
      `Agent.__init__`); add a per-key override such as
      `loss_scales.rec_fail`.
- [ ] **After env** Split the observation: put the variables the failure
      rules depend on in their own key (e.g. `vector_fail`), separate from
      the rest of `vector`.
- [ ] **Before env** Measure reconstruction error per key and failure-head
      recall, on all states and on near-failure states only (e.g. the last N
      steps before a failure).
- [ ] **Before env** Vector open-loop report. `Agent.report` decodes an
      observed first half and an imagined second half (`obsrecons`,
      `imgrecons`) but only logs image keys, so with vectors it logs nothing.
      Log instead, per vector key and imagined step, the prediction error in
      raw units (`nn.symexp` of `pred()`), and the failure head's recall on
      imagined features against the true `is_failure` labels.
- [ ] **After env** Only if those measurements show the model forgetting:
      lower `loss_scales.rep` or `dyn.rssm.free_nats`, or raise
      `dyn.rssm.classes`, and re-measure.

## 2. Rule-based failure check

- [ ] **Before env** Define the rule interface: a function from decoded
      observations (raw units) to a boolean, supplied per task.
- [ ] **After env** Write the actual failure rule for the real task.
- [ ] **Before env** In `Agent.loss`, run `self.dec` on the imagined features
      (`imgfeat`; `img` means imagined, not image) and apply `nn.symexp` to
      the predictions before applying the rule.
- [ ] **Before env** Combine: flagged if rule OR head.
- [ ] **Before env** Measure the extra compute of decoding in imagination;
      make it a config switch.

## 3. Failure-terminated imagination

- [x] In `Agent.loss`, compute the failure probability on the imagined
      features and pass `rew - fail_penalty * p_fail` and
      `con * (1 - p_fail)` into `imag_loss`. A failure ends the episode, so
      imagination stops at predicted failures.
- [ ] **Before env** Same in the replay value loss: `term | is_failure` and
      `rew - fail_penalty * is_failure`, using the real labels.
- [x] Add `fail_penalty` under `agent:` in `configs.yaml`.
- [ ] **After env** Size the penalty so it dominates: with Φ in [0, 1],
      `fail_penalty > shaping_scale + goal_bonus` (e.g. 1, 1 and 10), so no
      reward ever outweighs a failure.
- [ ] **After env** Check how the penalty interacts with return
      normalization (`retnorm` uses the 5-95 percentile range): watch
      `ret_min`, `ret_max` and `ret_rate`.
- [ ] **Before env** For the PPO baseline, write the env wrapper that adds
      the same penalty, used only by PPO.
- [ ] **After env** Metric: margin to the failure rule along each run (merge
      with the risk margin in section 6).

## 4. Ensemble uncertainty

- [ ] **Before env** Failure head ensemble: N heads with different
      initializations; aggregate with max (pessimistic).
- [ ] **Before env** Dynamics ensemble: N `_prior` networks in
      `dreamerv3/rssm.py`; decide which member `imagine` samples from.
- [ ] **Before env** Disagreement measure and penalty term in the imagined
      reward.
- [ ] **Before env** Config: ensemble size, disagreement scale.
- [ ] **After env** Tune ensemble size and disagreement scale.

## 5. Failure-focused replay

- [ ] **Before env** Choose the mechanism: the existing `Prioritized`
      selector (`embodied/core/selectors.py`) fed with a failure-based
      priority, or a dedicated selector that indexes sequences containing or
      approaching failure.
- [ ] **Before env** If `Prioritized`: emit `outs['replay']['priority']` with
      shape (B, T) from `Agent.train` (stub at the commented lines), and set
      `replay.fracs` in the config.
- [ ] **Before env** Correct for the sampling bias in the failure head and
      world model loss (importance weights), or accept and document the
      calibration shift.
- [ ] **After env** Optional: simulator resets into near-failure states.

## 6. Candidate export

- [ ] **Before env** Deterministic policy mode: `Agent.policy` ignores `mode`
      and always samples; add a mean/argmax path so a candidate is
      reproducible.
- [ ] **Before env** Candidate run: an eval-only script (based on
      `embodied/run/eval_only.py`) that runs the deterministic policy and
      records, per episode, the start condition / seed, the full action
      sequence, whether it failed, and the maximum progress reached.
- [ ] **Before env** Rank candidates:
  1. failure-free runs first, goal reached before goal not reached;
  2. failing runs by maximum progress (`log/progress`), then steps survived;
  3. within a group, by risk: largest predicted failure probability and
     smallest margin to a failure rule (pessimistic ensemble from section 4
     if available);
  4. task reward only breaks remaining ties.
- [ ] **After env** If start conditions vary, run each candidate on several
      of them and rank by failure rate across those runs first.
- [ ] **Before env** Always export the best candidates found, even if none is
      failure-free.
- [ ] **Before env** Keep several distinct candidates, so the exact
      simulation has alternatives if the top one fails.
- [ ] **Before env** Export format: start condition, action sequence,
      per-step predicted failure probability and margin, ensemble
      disagreement, and Nightmare's own simulator result.
- [ ] **Before env** Optional: screen extra candidates in imagination (no
      simulator calls) and send only the lowest-risk ones to the simulator.

## 7. Evaluation

- [ ] **After env** Main curve: best candidate against simulator steps used
      (failure-free or not, goal reached or not, otherwise maximum progress
      before the failure, or failure rate if start conditions vary).
- [ ] **After env** Simulator steps until the first failure-free candidate,
      if one is found.
- [ ] **After env** Agreement with the exact simulation: does Nightmare's
      outcome per candidate (failure or not, progress) match the exact one.
- [ ] **After env** Number of distinct candidates at the best rank, per
      budget.
- [ ] **After env** Baselines on the same budget: PPO, plain DreamerV3, and
      direct search (e.g. CEM or CMA-ES over action sequences) in the
      simulator.
- [ ] **After env** Ablations: each of sections 1-5 switched off
      individually.
- [ ] **After env** Only if patterns must hold across conditions: vary start
      conditions and measure the pass rate on held-out ones.

## Housekeeping

- [x] `Agent.report` dropped its loss metrics (`mets.update(mets)`).
- [x] `run.from_checkpoint_regex` was missing from the config.
- [ ] **Before env** `make_env` points `dm` at `embodied.envs.from_dmenv`;
      the file is `from_dm.py` (only matters for the `dm` suite).
- [ ] **Before env** `np.ranodm` typo in `embodied/core/streams.py`
      (`Mixer`, unused).
- [ ] **Before env** Update package name and Dockerfile for Nightmare.
- [ ] **Before env** README: rewrite for the exploration scope. Replace the
      deployment safety filter with candidate export, drop the Lagrangian
      constraint mode, the terminal-image paragraph and the domain
      randomization / sim-to-real roadmap item.
