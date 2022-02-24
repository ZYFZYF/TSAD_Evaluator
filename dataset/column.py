# @Time    : 2022/2/23 15:18
# @Author  : ZYF


class Column(object):
    def __init__(self, name: str, value_list: list[float]):
        super().__init__()
        self.name = name
        self.value_list = value_list

    def __getitem__(self, index):
        return self.value_list[index]


if __name__ == '__main__':
    col = Column("test", [x for x in range(10000)])
