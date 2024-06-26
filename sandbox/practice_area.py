import pandas as pd
import matplotlib.pyplot as plt

output_path = './src/models/results/model_report.csv'

file = './datasets/merge.csv'

data= pd.read_csv(file)

columns = data.iloc[:,3:].columns


for col in data[]:
    print(data[col])