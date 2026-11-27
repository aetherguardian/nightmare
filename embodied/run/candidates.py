# This file exists to be able to export patterns of agents that were able to learn failure-avoiding patterns.
# It is the main purpose and functinality of the Nightmare framework.

import collections
import json
from functools import parial as bind

import elements
import embodied
import numpy as np

def candidates(make_agent, make_env, args):
    asset args.from_checkpoint, 'Pass --run.from_checkpoint <path>'

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
        data = {f'act_{k}': np.stack([s[k] for s in steps[:-1]]) for k in act_keys}
        data.update({f'start_{k}': first[k] for k in obs_keys})
        data['progress'] = progress
        data['is_failure'] = np.array([s['is_failure'] for s in steps])
        np.savez_compressed(str(outdir / f'cand_{index:06d}.npz'), **data)
        summary.append({
            'index': index,
            'failed': failed,
            'goal': bool(last['is_terminal']) and not failed,
            'max_progress': float(np.nanmax(progress)) if np.isfinite(progress).any() else None,
            'length': len(steps),
            'seed': int(first['log/seed']) if 'log/seed' in first else None,
    })

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

    policy = lambda *a: agent.policy(*a, mode='eval')
    driver.reset(agent.init_policy)
    driver(policy, episodes=args.candidate_eps)
    driver.close()

    with open(str(outdir / 'summary.jsonl'), 'w') as f:
        for row in summary:
            f.write(json.dumps(row) + '\n')
    print(f'Saved {len(summary)} candidates to {outdir}')               

