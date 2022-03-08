# @Time    : 2022/3/7 18:48
# @Author  : ZYF


def get_meta_data(obj: object):
    return {k: v for k in dir(obj) if
            isinstance(v := obj.__getattribute__(k), (int, float, str)) and not k.startswith('__')}
