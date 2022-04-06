# TSAD_Evaluator

## TODO

    - 数据集接入
        - 一期五个 √
        - KPI出了问题，train数据没出来 √（不是问题，数据如此）
        - 做归一化

    - grafana
        - 展示单个序列结果dashboard  √
        - 展示单个数据集指标 √
        - 展示多个算法单个数据集指标  √
        - 展示多个算法单个序列结果 
    - 展示算法耗时 √

    - 实现方法
        - 以lstm和ae作为展示内容 √
        - 各种类型的方法代表
            - spot方法作为stream，显示上下阈值 √
            - 单维到多维 √
            - 集成 

    - 评估
        - 自动注册功能  √
        - point-adjusted 相关指标 √
        - NIPS 2018
        - TNNLR 2021
        - AAAI 2022
        - 触发式算法的评估

    - 打包成docker
        - 配置grafana
        - 配置influxdb