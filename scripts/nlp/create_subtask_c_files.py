import json


def jdefault(o):
    return o.__dict__


def extract_subtask_c(file_path='data/THF/sentence/subtaskB_v3_train.json',
                      output_path='data/THF/sentence/subtaskC_v3_train.json'):
    sentences = []
    with open(file_path, encoding='utf-8') as data_file:
        data = json.load(data_file)
        for sentence in data:
            if sentence['Label'] == 'ClaimPro' or sentence['Label'] == 'ClaimContra':
                sentences.append(sentence)
    print('{} sentences in {}'.format(len(sentences), output_path))
    with open(output_path, 'w') as outfile:
        json.dump(sentences, outfile, indent=2, default=jdefault)


if __name__ == '__main__':
    extract_subtask_c(file_path='data/THF/sentence/subtaskB_v3_train.json',
                      output_path='data/THF/sentence/subtaskC_v3_train.json')
    extract_subtask_c(file_path='data/THF/sentence/subtaskB_v3_test.json',
                      output_path='data/THF/sentence/subtaskC_v3_test.json')
