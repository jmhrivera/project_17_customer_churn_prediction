import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('./datasets/merge.csv')

data.columns

plt.hist(data['SeniorCitizen'])
plt.show()