import json

if __name__ == "__main__":
    new_data = []
    with open('./data/train_preprocessed/trainset/search.train.json') as fin:
        for idx, line in enumerate(fin):
            json_obj = json.loads(line.strip())
            new_data.append(json_obj)
            if idx > 10000:
                break
    with open('./data/train_preprocessed/trainset/search.train.cut.json', 'a') as fout:
        for line in new_data:
            fout.write(json.dumps(line)+'\n')
