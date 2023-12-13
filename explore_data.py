import pandas as pd

file = './archive/cybersecurity_attacks.csv'
file2 = './archive/cicddos2019_dataset.csv'
data = pd.read_csv(file)

print(data.columns)

drop_list = ['IDS/IPS Alerts','Log Source', 'Firewall Logs', 'Network Segment', 'Geo-location Data', 'Action Taken',
             'User Information', 'Attack Signature', 'Attack Type', 'Alerts/Warnings', 'Malware Indicators', 'Anomaly Scores',
             'Severity Level', 'Proxy Information']
drop_list.extend(['Device Information', 'Timestamp', 'Payload Data'])
ddos_data = data[data['Attack Type'] == 'DDoS'].drop(drop_list, axis=1)
print(ddos_data.head())

data2 = pd.read_csv(file2)
print(data2.columns)