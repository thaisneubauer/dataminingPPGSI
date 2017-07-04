import pandas as pd

def open_file(path, filename):
    return pd.read_csv(path + filename, header=None)


def parse_column_name(line):
    column_number = int(line.split('.')[0]) - 1
    rest_of_line = line.split('.')[1]
    column_name = rest_of_line.split(':')[0].strip()
    return {column_number: column_name}


def columns_names_dict(path, filename):
    columns_dict = {}
    count = 0
    with open(path + filename) as f:
        for line in f.readlines():
            line_result = parse_column_name(line)
            columns_dict.update(line_result)
            count += 1
    columns_dict.update({(count): 'label'})
    return columns_dict


def get_data():
    path = 'data/classification/'
    data = open_file(path, 'cars.csv')
    columns_dict = columns_names_dict(path, 'cars_columns.csv')
    data = data.rename(columns=columns_dict)
    return data