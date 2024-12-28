import pandas as pd
import numpy as np

# Generate Simple
np.random.seed(42)

X = np.random.rand(100, 1) # features
y = 2 * X + 1 + np.random.normal(0, 0.1, size = (100, 1))

df = pd.DataFrame({ 'X': X.flatten(), 'y' : y.flatten() })
df.to_csv('data.csv', index=False)