import json
import os

with open('train.json', 'r') as json_file:
    obj = json.load(json_file)

    parsed_output = []

    for title in obj['data']:
        for paragraph in title['paragraphs']:
            parsed_output.append(paragraph)

    with open('clean_train.txt', 'w', encoding='utf-8') as outfile:
        for line in parsed_output:
            outfile.write(str(line) + '\n')
