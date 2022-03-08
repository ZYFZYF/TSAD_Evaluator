# @Time    : 2022/2/24 13:57
# @Author  : ZYF
import pandas as pd
from influxdb import InfluxDBClient, DataFrameClient

from config import INFLUXDB_HOST, INFLUXDB_PORT, INFLUXDB_USERNAME, INFLUXDB_PASSWORD, INFLUXDB_DBNAME

client = InfluxDBClient(host=INFLUXDB_HOST,
                        port=INFLUXDB_PORT,
                        username=INFLUXDB_USERNAME,
                        password=INFLUXDB_PASSWORD,
                        database=INFLUXDB_DBNAME)
if INFLUXDB_DBNAME not in client.get_list_database():
    client.create_database(dbname=INFLUXDB_DBNAME)
dataframe_client = DataFrameClient(host=INFLUXDB_HOST,
                                   port=INFLUXDB_PORT,
                                   username=INFLUXDB_USERNAME,
                                   password=INFLUXDB_PASSWORD,
                                   database=INFLUXDB_DBNAME)


def save_ts_data(data: pd.DataFrame, table_name: str):
    dataframe_client.drop_measurement(measurement=table_name)
    dataframe_client.write_points(dataframe=data, measurement=table_name, batch_size=5000)


def load_ts_data(table_name: str):
    return dataframe_client.query(f'select * from \"{table_name}\"')[table_name]


def save_meta_data(measurement: str, tags: dict, fields: dict):
    client.write_points(points=[{'measurement': measurement,
                                 'time': 0,
                                 'tags': tags,
                                 'fields': fields}])


def get_where_clause_from_tags(tags: dict):
    return 'and'.join([f'\"{k}\"=\'{v}\'' for k, v in tags.items()])


def load_meta_data(measurement: str, tags: dict):
    return next(client.query(f'select * from \"{measurement}\" where {get_where_clause_from_tags(tags)}').get_points(
        measurement=measurement), {})


def query_field_of_tags(measurement: str, tags: dict, field: str):
    return [item[field] for item in
            client.query(f'select * from \"{measurement}\" where {get_where_clause_from_tags(tags)}').get_points(
                measurement=measurement)]


if __name__ == '__main__':
    pass
