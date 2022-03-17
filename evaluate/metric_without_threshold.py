# @Time    : 2022/3/10 23:38
# @Author  : ZYF
import abc

metric_without_threshold_list = []


class MetricWithoutThreshold(abc.ABCMeta):

    def __init_subclass__(mcs, **kwargs):
        metric_without_threshold_list.append(mcs)

    @classmethod
    @abc.abstractmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        ...


class BestF1(MetricWithoutThreshold):
    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        best_f1_score = 0
        n = len(label)
        order = [i for i in range(n)]
        order = sorted(order, key=lambda x: predict[x], reverse=True)
        TP = 0
        TN = n - sum(label)
        FN = sum(label)
        FP = 0
        for i in range(n):
            post = order[i]
            if label[post] == 0:
                TN -= 1
                FP += 1
            else:
                TP += 1
                FN -= 1
            recall = 1.0 * TP / (TP + FN)
            precision = 1.0 * TP / (TP + FP)
            f1_score = 2.0 * recall * precision / (recall + precision + 1e-10)
            if f1_score > best_f1_score:
                best_f1_score = f1_score
        return best_f1_score


class BestFpa(MetricWithoutThreshold):

    @classmethod
    def score(mcs, predict: list[float], label: list[float]) -> float:
        best_f1_score = 0
        n = len(label)
        head = [0] * n
        tail = [0] * n
        le = 0
        while le < n:
            ri = le
            while ri < n and label[ri] == 1:
                ri += 1
            for i in range(le, ri):
                head[i] = le
                tail[i] = ri
            le = ri + 1
        order = [i for i in range(n)]
        order = sorted(order, key=lambda x: predict[x], reverse=True)
        pred = [0] * n
        TP = 0
        TN = n - sum(label)
        FN = sum(label)
        FP = 0
        for i in range(n):
            post = order[i]
            if pred[post] == 0:
                if label[post] == 0:
                    pred[post] = 1
                    TN -= 1
                    FP += 1
                else:
                    for j in range(head[post], tail[post]):
                        if pred[j] == 0:
                            pred[j] = 1
                            TP += 1
                            FN -= 1
                        else:
                            break
            recall = 1.0 * TP / (TP + FN)
            precision = 1.0 * TP / (TP + FP)
            f1_score = 2.0 * recall * precision / (recall + precision + 1e-10)
            if f1_score > best_f1_score:
                best_f1_score = f1_score
        return best_f1_score


if __name__ == '__main__':
    x = [0, 0.2, 0.3, 0.5, 0.6, 0.1]
    y = [0, 1, 1, 1, 0, 1]
    print(metric_without_threshold_list)
    for metric in metric_without_threshold_list:
        print(metric.__name__, metric.score(x, y))
