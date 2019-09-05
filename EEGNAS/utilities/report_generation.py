import os
from collections import defaultdict
import re
import pandas as pd
from EEGNAS import global_vars


def get_base_folder_name(fold_names, first_dataset):
    ind = fold_names[0].find('_')
    end_ind = fold_names[0].rfind(first_dataset)
    base_folder_name = list(fold_names[0])
    base_folder_name[ind + 1] = 'x'
    base_folder_name = base_folder_name[:end_ind - 1]
    base_folder_name = ''.join(base_folder_name)
    base_folder_name = add_params_to_name(base_folder_name, global_vars.get('include_params_folder_name'))
    return base_folder_name


def generate_report(filename, report_filename):
    params = ['final', 'from_file']
    params_to_average = defaultdict(float)
    avg_count = defaultdict(int)
    data = pd.read_csv(filename)
    for param in params:
        for index, row in data.iterrows():
            if param in row['param_name'] and 'raw' not in row['param_name'] and 'target' not in row['param_name']:
                row_param = row['param_name']
                intro = re.compile('\d_')
                if intro.match(row_param):
                    row_param = row_param[2:]
                outro = row_param.find('from_file')
                if outro != -1:
                    row_param = row_param[outro:]
                params_to_average[row_param] += float(row['param_value'])
                avg_count[row_param] += 1
    for key, value in params_to_average.items():
        params_to_average[key] = params_to_average[key] / avg_count[key]
    pd.DataFrame(params_to_average, index=[0]).to_csv(report_filename)


def concat_and_pivot_results(fold_names, first_dataset):
    to_concat = []
    for folder in fold_names:
        full_folder = 'results/' + folder
        files = [f for f in os.listdir(full_folder) if os.path.isfile(os.path.join(full_folder, f))]
        for file in files:
            if file[0].isdigit():
                to_concat.append(os.path.join(full_folder, file))
    combined_csv = pd.concat([pd.read_csv(f) for f in to_concat])
    pivot_df = combined_csv.pivot_table(values='param_value',
                              index=['exp_name', 'machine', 'dataset', 'date', 'generation', 'subject', 'model'],
                              columns='param_name', aggfunc='first')
    filename = f'{get_base_folder_name(fold_names, first_dataset)}_pivoted.csv'
    pivot_df.to_csv(filename)
    return filename


def add_params_to_name(exp_name, multiple_values):
    if multiple_values:
        for mul_val in multiple_values:
            exp_name += f'_{mul_val}_{global_vars.get(mul_val)}'
    return exp_name
