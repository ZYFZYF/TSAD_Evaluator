# @Time    : 2022/2/24 13:57
# @Author  : ZYF
from influxdb import InfluxDBClient, DataFrameClient

from database.databasetool import DatabaseTool
from dataset.dataset import Dataset
from dataset.time_series import TimeSeries
from utils.config import CONFIG


class InfluxdbTool(DatabaseTool):
    host = CONFIG.get('influxdb').get('host')
    port = CONFIG.get('influxdb').get('port')
    username = CONFIG.get('influxdb').get('username')
    password = CONFIG.get('influxdb').get('password')
    dbname = CONFIG.get('influxdb').get('dbname')
    client = InfluxDBClient(host, port, username, password, dbname)
    if dbname not in client.get_list_database():
        client.create_database(dbname=dbname)
    dataframe_client = DataFrameClient(host, port, username, password, dbname)

    @classmethod
    def load_time_series(cls, name) -> TimeSeries:
        return TimeSeries(cls.dataframe_client.query(f'select * from {name}')[name])

    @classmethod
    def load_dataset(cls, name):
        dataset = Dataset(name)
        for measurement in cls.client.get_list_measurements():
            if (res := cls.generate_time_series_name_and_dataset_name_from_measurement(measurement))[0] == name:
                dataset[res[1]] = cls.load_time_series(measurement)

    @classmethod
    def save_time_series(cls, name, time_series: TimeSeries):
        cls.dataframe_client.write_points(time_series, name)

    @classmethod
    def save_dataset(cls, dataset: Dataset):
        for time_series_name, time_series in dataset.items():
            cls.save_time_series(cls.generate_time_series_table_name(dataset.name, time_series_name), time_series)

    @classmethod
    def generate_time_series_table_name(cls, dataset_name, time_series_name):
        return dataset_name + '@' + time_series_name

    @classmethod
    def generate_time_series_name_and_dataset_name_from_measurement(cls, measurement):
        return measurement.split('@')


if __name__ == '__main__':
    client = InfluxdbTool()
    df = TimeSeries(data={'a': [1, 2, 3], 'b': [4, 5, 6], 'label': [0, 0, 0]})
    print(df)
    print(df.index)
    # df.index = pd.DatetimeIndex(df.index)
    print(df)
    client.dataframe_client.write_points(df, measurement='demo')
    print(client.dataframe_client.query("select * from demo"))
    x = client.dataframe_client.query("select * from demo")
    y = x['demo']
    print(y, y.index)
    client.dataframe_client.drop_measurement('demo')
