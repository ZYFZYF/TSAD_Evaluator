# @Time    : 2022/3/7 18:48
# @Author  : ZYF

import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


def get_meta_data(obj: object):
    return {k: v for k in dir(obj) if
            isinstance(v := obj.__getattribute__(k), (int, float, str)) and not k.startswith('__')}
