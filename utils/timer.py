# @Time    : 2022/3/17 19:54
# @Author  : ZYF
import time
from collections import defaultdict

cost_time = defaultdict(float)


def timer(stage):
    def func(fun):
        def wrapper(*args, **kwargs):
            time_start = time.time()
            result = fun(*args, **kwargs)
            cost_time[stage] += time.time() - time_start
            return result

        return wrapper

    return func


def time_on():
    cost_time.clear()


def get_time():
    return dict(cost_time)
