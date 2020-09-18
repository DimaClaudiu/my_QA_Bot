import ast
import json

MAX_SEQ_LEN = 256

with open('classes.json') as file:
    obj = json.load(file)
    classes = obj.keys()

for cls in classes:
    with open(f'categories/{cls}.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

        contexts = []
        for line in lines:
            obj = ast.literal_eval(line)
            context = obj['headline'] + " " + obj['short_description']
            if len(context) > 1:
                contexts.append({'text': context, 'date': obj['date']})

    with open(f'split_contexts/{cls}.txt', 'w', encoding='utf-8') as file:
        for context in contexts:
            file.write(str(context) + '\n')
