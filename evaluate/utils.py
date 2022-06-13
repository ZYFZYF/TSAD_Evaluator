# @Time    : 2022/5/7 10:32
# @Author  : ZYF
def safeFloatDivide(a, b):
    if b == 0:
        return 0
    else:
        return 1.0 * a / b


def getPRF(TP, FP, FN) -> tuple[float, float, float]:
    precision = safeFloatDivide(TP, TP + FP)
    recall = safeFloatDivide(TP, TP + FN)
    f1_score = safeFloatDivide(recall * precision * 2, recall + precision)
    return precision, recall, f1_score


def extractEvent(label: list[float]) -> tuple[list[int], list[int], int]:
    n = len(label)
    count = 0
    le = 0
    head = [-1] * n
    tail = [-1] * n
    while le < n:
        while le < n and label[le] != 1:
            le += 1
        if le == n:
            break
        count += 1
        ri = le
        while ri + 1 < n and label[ri + 1] == 1:
            ri += 1
        for i in range(le, ri + 1):
            head[i] = le
            tail[i] = ri
        le = ri + 1
    return head, tail, count
