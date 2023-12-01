import pandas as pd

file = './archive/cybersecurity_attacks.csv'
data = pd.read_csv(file)

print(data.columns)
print(data.dtypes)
print(data['Attack Type'].unique())

print(data['Malware Indicators'].head())
