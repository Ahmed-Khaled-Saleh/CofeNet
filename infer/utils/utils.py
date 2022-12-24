import json

def read_write_str(input_str_lst : list, path_to_input_file: str):
    with open(path_to_input_file, 'w') as file:
        for item in input_str_lst:
            lst_items = item.split()
            to_write = {'tokens': lst_items}
            file.write(json.dumps(to_write) + '\n')
            
        return path_to_input_file

def load_text_file_by_line(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        data = [token.replace('\n', '').replace('\r', '') for token in f.readlines()]
    return [x for x in data if x]


def load_json_file_by_line(file_path):
    return [json.loads(line) for line in load_text_file_by_line(file_path)]


def load_data_from_file(file_path):
    data = []
    for line in load_json_file_by_line(file_path):
        dict_inst = {'tokens': [w.lower() for w in line['tokens']]}
        data.append(dict_inst)
    return data
