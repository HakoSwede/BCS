import os

import numpy as np
import pandas as pd

sharpe_df = pd.read_csv('datasets/heatmap.csv', index_col=0)
sum_df = pd.DataFrame(
    data=np.zeros_like(sharpe_df.values),
    index=sharpe_df.index,
    columns=sharpe_df.columns
)

for r in [-1, 0, 1]:
    for c in [-1, 0, 1]:
        sum_df += sharpe_df.shift(r, axis=0).shift(c, axis=1).fillna(0)

for r in [0, -1]:
    for c in [0, -1]:
        sum_df.iloc[r, c] /= 4

sum_df.iloc[1:-2, 0] /= 6
sum_df.iloc[1:-2, -1] /= 6
sum_df.iloc[0, 1:-2] /= 6
sum_df.iloc[-1, 1:-2] /= 6
sum_df.iloc[1:-2, 1:-2] /= 9
