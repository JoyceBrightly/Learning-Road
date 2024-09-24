import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sympy.physics.units.definitions.dimension_definitions import information

data = load_iris()

datatype = type(data)
# print(datatype)

keys = data.keys()
# print(keys)

sample_data = data.data
# print(sample_data)

target = data.target
# print(target)

labels = data.feature_names
# print(labels)

df = pd.DataFrame(data = sample_data, columns = labels)
df['target'] = target
# print(df)

# datadfinfo = df.info()
# print(datadfinfo)

datadfdescribe = df.describe()
# print(datadfdescribe)

Boxplot = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].plot(kind = 'box',
                                                                                                      subplots = True,
                                                                                                      layout = (2,2),
                                                                                                      sharex=False,
                                                                                                      sharey=False)
print(Boxplot)

# df.loc[:, ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)','target']].corr()


import seaborn as sb
pairplot = sb.pairplot(df, hue = 'target')
print(pairplot)
