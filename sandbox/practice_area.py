import pandas as pd
import matplotlib.pyplot as plt
describe = pd.read_csv('./src/eda/files/describe.csv')
info = pd.read_csv('./src/eda/files/info.csv')

pd.DataFrame(describe.info())


