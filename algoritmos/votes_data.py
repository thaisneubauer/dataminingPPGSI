import pandas as pd

def open_file(path, filename):
    return pd.read_csv(path + filename, header=None)

def parse_column(line):
    column_number = int(line.split('.')[0])-1
    rest_of_line = line.split('.')[1]
    column_name = rest_of_line.split(':')[0].strip()
    return {column_number: column_name}, {column_name:{'y':str(column_name)+'-y', 'n':str(column_name)+'-n', '?':str(column_name)+'-?'}}


def get_dicts(path, filename):
    columns_dict = {}
    values_dict = {}
    with open(path + filename) as f:
        for line in f.readlines():
            columns, values = parse_column(line)
            columns_dict.update(columns)
            values_dict.update(values)
    return columns_dict, values_dict


def get_data():
    path = 'data/association/'
    data = open_file(path, 'house-votes-84.csv')
    columns_dict, values_dict = get_dicts(path, 'house-votes-84_columns.csv')
    data = data.rename(columns=columns_dict)
    data = data.replace(values_dict)
    return data