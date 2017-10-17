from argminingdeeplearning.loaders import vocabulary_builder

if __name__ == '__main__':
    file_path = 'data/THF/sentence/subtaskA_v3_train.json'
    vocabulary_builder.create_mappings(file_path)