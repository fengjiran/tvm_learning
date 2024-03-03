import numpy as np
import pandas as pd

df = pd.read_excel('data.xlsx')
# print(df)

# pre = df.values[2][1:]
# real = df.values[3][1:]
pre = df['Unnamed: 2'][12 : 12 + 96]
real = df['Unnamed: 3'][12 : 12 + 96]
np.testing.assert_equal(pre.shape, real.shape)

RMSE = np.mean(np.square(real - pre)) ** 0.5
MAPE = np.mean(np.abs(np.divide(real - pre, real)))

print("RMSE: {}".format(RMSE))
print("MAPE: {}".format(MAPE))

# a = np.array([1,2])
# b = np.array([2,4])
# c = np.divide(a, b)
# print()
