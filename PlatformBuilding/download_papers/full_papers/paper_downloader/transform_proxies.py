import pandas as pd
import json
import pickle

# proxies = pd.read_csv('http_socks5_socks4_proxies.csv')
# print(proxies.columns)
#
# proxies_list = []
# for i in range(len(proxies)):
#     row = proxies.iloc[i, :]
#     if row['protocol'] == 'http':
#         proxies_list.append('http://{}:{}'.format(row['ip'], row['port']))
#
# with open('proxies.txt', 'w') as f:
#     f.writelines('\n'.join(proxies_list))
#
#
# with open('proxies.json', 'w') as f:
#     json.dump(proxies_list, f)

# proxies = pd.read_csv('Free_Proxy_List.csv')
# proxies_list = []
# for i in range(len(proxies)):
#     row = proxies.iloc[i, :]
#     proxies_list.append('{}://{}:{}'.format(row['protocols'], row['ip'], row['port']))
#
# print(proxies_list)
# with open('proxies.txt', 'w') as f:
#     f.writelines('\n'.join(proxies_list))

with open('http.txt', 'r') as f:
    lines = f.readlines()


proxy_list = []
with open('proxies.txt', 'w') as f2:
    for line in lines:
        to_save = 'https://{}'.format(line)
        f2.write(to_save)
        proxy_list.append(to_save.strip())


print(proxy_list)
