import json
import pathlib, os
import warnings

import numpy as np


def load_runs(filename):
  runs = []
  with open(filename) as f:
    for line in f:
      runs.append(json.loads(line))
  return runs

def compute_success_rates(runs, sortby=None):
  tasks = sorted(key for key in runs[0] if key.startswith('achievement_'))
  percents = np.empty(len(tasks))
  percents[:] = 0
  for run in runs:
    for key, values in run.items():
      if key in tasks:
        k = tasks.index(key)
        percent = 1 if values > 0 else 0
        percents[k] = percents[k] + percent
  percents = percents/len(runs)
  percents = np.round(percents, 4)
  order = np.argsort(-np.nanmean(percents[sortby], 0), -1)
  percents = percents[order]
  percents = percents*100
  tasks = np.array(tasks)[order].tolist()
  return percents, tasks

def compute_achievements(traj):
  tasks = sorted(key for key in traj if key.startswith('achivement_'))
  numbers = np.empty(len(tasks))
  numbers[:] = 0
  for key, values in traj.items():
    if key in tasks:
      k = tasks.index(key)
      numbers[k] = np.max(values)
  tasks = np.array(tasks).tolist()
  return numbers, tasks


def compute_scores(percents):
  # Geometric mean with an offset of 1%.
  assert (0 <= percents).all() and (percents <= 100).all()
  if (percents <= 1.0).all():
    print('Warning: The input may not be in the right range.')
  with warnings.catch_warnings():  # Empty seeds become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
  return scores


def binning(xs, ys, borders, reducer=np.nanmean, fill='nan'):
  xs, ys = np.array(xs), np.array(ys)
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  binned = []
  with warnings.catch_warnings():  # Empty buckets become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    for start, stop in zip(borders[:-1], borders[1:]):
      left = (xs <= start).sum()
      right = (xs <= stop).sum()
      if left < right:
        value = reducer(ys[left:right])
      elif binned:
        value = {'nan': np.nan, 'last': binned[-1]}[fill]
      else:
        value = np.nan
      binned.append(value)
  return borders[1:], np.array(binned)

def read_trajectories(dir_path):
  traj_npz = [x for x in os.listdir(dir_path) if x.endswith(".npz")]
  data = dict(np.load(os.path.join(dir_path, traj_npz[0]), allow_pickle=True))
  traj = {key: data[key] for key in data.keys()}
  rates, tasks = compute_achievements(traj)
  print("Tasks:")
  print(tasks)
  for el in traj_npz:
    data = dict(np.load(os.path.join(dir_path, el), allow_pickle=True))
    traj = {key: data[key] for key in data.keys()}
    print(str(el))
    rates, tasks = compute_achievements(traj)
    print("Length: " + str(len(traj["reward"])))
    print(rates)

if __name__ == "__main__":
  runs = load_runs("recording/stats.jsonl")
  rates, tasks = compute_success_rates(runs)
  print("Tasks:")
  print(tasks)
  print("Success rates:")
  print(rates)
  scores = compute_scores(rates)
  print("Score:" + str(round(scores, 3)))
  #read_trajectories("dreamerv3/dataset")