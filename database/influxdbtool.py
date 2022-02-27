# @Time    : 2022/2/24 13:57
# @Author  : ZYF
import pandas as pd
from influxdb import InfluxDBClient, DataFrameClient

from utils.config import CONFIG

host = CONFIG.get('influxdb').get('host')
port = CONFIG.get('influxdb').get('port')
username = CONFIG.get('influxdb').get('username')
password = CONFIG.get('influxdb').get('password')
dbname = CONFIG.get('influxdb').get('dbname')
client = InfluxDBClient(host, port, username, password, dbname)
if dbname not in client.get_list_database():
    client.create_database(dbname=dbname)
dataframe_client = DataFrameClient(host, port, username, password, dbname)


def save_ts_data(data: pd.DataFrame, table_name: str):
    dataframe_client.write_points(dataframe=data, measurement=table_name, batch_size=1000)


def load_ts_data(table_name: str):
    return dataframe_client.query(f'select * from \"{table_name}\"')[table_name]


def save_meta_data(tags: dict, fields: dict):
    client.write_points(points=[{'measurement': 'ts_meta',
                                 'time': 0,
                                 'tags': tags,
                                 'fields': fields}])


def get_where_clause_from_tags(tags: dict):
    return 'and'.join([f'\"{k}\"=\'{v}\'' for k, v in tags.items()])


def load_meta_data(tags: dict):
    return next(client.query(f'select * from ts_meta where {get_where_clause_from_tags(tags)}').get_points('ts_meta'),
                {})


def query_field_of_tags(tags: dict, field: str):
    return [item[field] for item in
            client.query(f'select * from ts_meta where {get_where_clause_from_tags(tags)}').get_points('ts_meta')]


if __name__ == '__main__':
    pass
