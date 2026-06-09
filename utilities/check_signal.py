import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/1_timeseries_canonical.csv")

for col in df.columns[:4]:
    plt.figure()
    plt.plot(df[col])
    plt.title(col)

plt.show()