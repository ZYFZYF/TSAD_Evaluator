# @Time    : 2022/2/24 14:16
# @Author  : ZYF

import yaml

CONFIG = None
with open('../config.yml') as f:
    CONFIG = yaml.safe_load(f.read())
