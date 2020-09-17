import json
import collections
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import random


def clean_text(text, max_len=128, stem=False):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    words = [word for word in word_tokens if word.isalpha()]

    filtered_words = [w for w in words if not w in stop_words]

    if stem:
        porter = PorterStemmer()
        filtered_words = [porter.stem(word) for word in filtered_words]

    return ' '.join(filtered_words)[0:max_len-1].lower()


def main():

    file = open('news_dataset.json', 'r', encoding='utf-8')
    lines = file.readlines()

    random.shuffle(lines)

    categories = {}

    for line in lines:
        obj = json.loads(line)

        if obj['category'] in categories:
            categories[obj['category']] += 1
        else:
            categories[obj['category']] = 0

    sorted_x = sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
    ordered = collections.OrderedDict(sorted_x)

    n = 10
    n_items = {key: value for key, value in list(ordered.items())[0:n]}

    mappings = {}
    i = 0
    for category, count in n_items.items():
        print(category + ": " + str(count))
        mappings[category] = i
        i += 1

    print(mappings)

    data = []

    c = 0
    for line in lines:
        obj = json.loads(line)

        category = obj['category']

        headline = clean_text(obj['headline'])
        content = clean_text(obj['short_description'])

        if obj['category'] in n_items:
            if len(headline) > 1:
                data.append([headline, mappings[category]])
            if len(content) > 1:
                data.append([content, mappings[category]])

        c += 1
        if c % 100 == 0:
            print(c)

    df = pd.DataFrame(data, columns=['text', 'labels'])

    print(df)

    df.to_csv('news2.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
