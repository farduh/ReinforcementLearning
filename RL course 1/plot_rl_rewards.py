import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
args = parser.parse_args()

a = np.load(f'linear_rl_trader_rewards_1/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

if args.mode == 'train':
  # show the training progress
  plt.plot(a)
else:
  # test - show a histogram of rewards
  plt.hist(a, bins=30)
  plt.show()
  df = pd.read_csv('linear_rl_trader_rewards_1/portfolio_evolution.csv')
  df.multiply(1/df.iloc[0]).plot()
  
plt.title(args.mode)
plt.show()