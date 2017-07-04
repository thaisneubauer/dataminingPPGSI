import pandas as pd

def open_file(path, filename):
    return pd.read_csv(path + filename, header=None)


def conversion_string_to_dict(conversion):
    each_conversion = conversion.split(',')
    result = {}
    for conv in each_conversion:
        name, abb = conv.split('=')
        result.update({abb: name})
    return result


def parse_line(line):
    cleaned_line = line.split('.')[1]
    column_name = cleaned_line.split(':')[0].strip()
    conversion = cleaned_line.split(':')[1].strip()
    conversion_dict = conversion_string_to_dict(conversion)
    return {column_name: conversion_dict}
 
    
def columns_values_dict(path, filename):
    dictionary = {}
    dictionary.update({'label':{'p':'poisonous', 'e':'edible'}})
    with open(path + filename) as f:
        for line in f.readlines():
            line_result = parse_line(line)
            dictionary.update(line_result)
    return dictionary


def parse_column_name(line):
    column_number = int(line.split('.')[0])
    rest_of_line = line.split('.')[1]
    column_name = rest_of_line.split(':')[0].strip()
    return {column_number: column_name}


def columns_names_dict(path, filename):
    columns_dict = {}
    with open(path + filename) as f:
        columns_dict.update({0: 'label'})
        for line in f.readlines():
            line_result = parse_column_name(line)
            columns_dict.update(line_result)
    return columns_dict


def get_data():
    path = 'data/association/'
    data = open_file(path, 'mushrooms.csv')
    columns_dict = columns_names_dict(path, 'mushrooms_columns.csv')
    data = data.rename(columns=columns_dict)
    values_dict = columns_values_dict(path, 'mushrooms_columns.csv')
    data = data.replace(values_dict)
    return data