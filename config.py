# @Time    : 2022/3/8 09:40
# @Author  : ZYF
# about influxdb connection
INFLUXDB_HOST = 'localhost'
INFLUXDB_PORT = '8086'
INFLUXDB_USERNAME = 'root'
INFLUXDB_PASSWORD = 'root'
INFLUXDB_DBNAME = 'TSAD_Evaluator'

# about dataset split
DATASET_TEST_SPLIT = 0.5

# about influxdb measurement name
RAW_TIME_SERIES_MEASUREMENT = 'ts_meta'
RESULT_TIME_SERIES_MEASUREMENT = 'result_ts_meta'
DETECTOR_MEASUREMENT = 'detector_meta'
DELIMITER = '@'

# about detector result columns name
ANOMALY_SCORE_COLUMN = 'score'
THRESHOLD_COLUMN = 'threshold'
LABEL_COLUMN = 'label'

# about input of time series
DATETIME_COLUMN_NAMES = ['datetime']
TIMESTAMP_COLUMN_NAMES = ['ts', 'timestamp']
LABEL_COLUMN_NAMES = ['label', 'anomaly', 'is_anomaly']

# about field of result
TRAIN_TIME = 'TrainTime'
TEST_TIME = 'TestTime'

# about some math constants
EPS = 1e-10
