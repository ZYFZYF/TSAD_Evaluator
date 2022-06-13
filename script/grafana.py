# @Time    : 2022/4/8 10:48
# @Author  : ZYF

import requests

GRAFANA_URL = 'http://166.111.68.233:43235'
dashboard_uids = ['cHnqjkLnz', 'LRcc38Lnk', 'G7kyIQP7z', 'QnmP-PEnk', 'zdiHqhE7k']
for uid in dashboard_uids:
    headers = {'Accept': 'application/json',
               'Content-Type': 'application/json'}
    res = requests.get(url=f'{GRAFANA_URL}/api/dashboards/uid/{uid}', headers=headers)
    print(res)
    print(res.data)
    print(res.headers)
    print(res.params)
