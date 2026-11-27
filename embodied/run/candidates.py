# This file exists to be able to export patterns of agents that were able to learn failure-avoiding patterns.
# It is the main purpose and functinality of the Nightmare framework.

import collections
import json
from functools import partial as bind

import elements
import embodied
import numpy as np

def candidates(make_agent, make_env, args):
    assert args.from_checkpoint, 'Pass --run.from_checkpoint <path>'

    agent = make_agent()
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(args.from_checkpoint, keys=['agent'])

    outdir = elements.Path(args.logdir) / 'candidates'
    outdir.mkdir()
    act_keys = list(agent.act_space.keys())
    obs_keys = list(agent.obs_space.keys())
    episodes = collections.defaultdict(list)
    summary = []

    env0 = make_env(0)
    raw = env0
    while hasattr(raw, 'env'):
      raw = raw.env
    bounds = {k: raw.act_space[k] for k in act_keys}
    env0.close()

    def save(steps):
        if len(steps) < 2:
            return
        index = len(summary)
        first, last = steps[0], steps[-1]
        failed = any(bool(s['is_failure']) for s in steps)
        progress = np.array(
            [s.get('log/progress', np.nan) for s in steps], np.float32)
        # The action stored at step t is executed at step t + 1, so the action
        # stored on the last step is never executed.
        data = {f'act_{k}': to_sim_units(k, np.stack([s[k] for s in steps[:-1]]))
                for k in act_keys}
        data.update({f'start_{k}': first[k] for k in obs_keys})
        data['progress'] = progress
        data['is_failure'] = np.array([s['is_failure'] for s in steps])
        data['fail_risk'] = np.array([s['fail_risk'] for s in steps], np.float32)
        data['reward'] = np.array([s['reward'] for s in steps], np.float32)
        np.savez_compressed(str(outdir / f'cand_{index:06d}.npz'), **data)
        summary.append({
            'index': index,
            'failed': failed,
            'goal': bool(last['is_terminal']) and not failed,
            'max_progress': float(np.nanmax(progress)) if np.isfinite(progress).any() else None,
            'max_pfail': float(max(s['fail_risk'] for s in steps)),
            'length': len(steps),
            'seed': int(first['log/seed']) if 'log/seed' in first else None,
            'score': float(sum(s['reward'] for s in steps)),
    })

    # Retrieve actual raw input units for pattern replication, not converted units
    def to_sim_units(key, a):
      space = bounds[key]
      if space.discrete:
        return a
      low = np.asarray(space.low, np.float32)
      high = np.asarray(space.high, np.float32)
      finite = np.isfinite(low) & np.isfinite(high)
      a = np.clip(a, -1, 1)
      real = (a + 1) / 2 * (high - low) + low
      return np.where(finite, real, a)

    def record(tran, worker):
        steps = episodes[worker]
        if tran['is_first']:
            steps.clear()
        steps.append({k: np.array(v) for k, v in tran.items()})
        if tran['is_last']:
            save(steps)
            steps.clear()

    fns = [bind(make_env, i) for i in range(args.envs)] 
    driver = embodied.Driver(fns, parallel=not args.debug)
    driver.on_step(record)

    policy = lambda *a: agent.policy(*a, mode='export')
    driver.reset(agent.init_policy)
    driver(policy, episodes=args.candidate_eps)
    driver.close()

    with open(str(outdir / 'summary.jsonl'), 'w') as f:
        for row in summary:
            f.write(json.dumps(row) + '\n')
    print(f'Saved {len(summary)} candidates to {outdir}')               

