from datasets import load_dataset, ClassLable
from transformers import AutoTokenizer

from configuration import config



def process():
    # 读取数据
    dataset_dict = load_dataset(
        'csv',data_files={
            'train':str(config.RAW_DATA_DIR / 'train.txt'),
            'vaild':str(config.RAW_DATA_DIR / 'vaild.txt'),
            'test':str(config.RAW_DATA_DIR / 'test.txt'),
        },delimiter = '\t'
    )

    # 数据清洗
    dataset_dict = dataset_dict.filter(lambda x:x['label'] is not None and x['text_a'] is not None)
    # 数据划分
    all_lables = sort(set(dataset_dict['train']['label']))
    dataset_dict = dataset_dict.cast_column('lable',ClassLable(all_lables))

    # 保存lables
    with open(config.PROCESSED_DATA_DIR / 'lables.json','w') as f:
        f.write('\n'.join(all_lables))

    # 数据集分词
        # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
    def tokenize(batch):
        inputs = tokenizer(batch(text_a),truncation = True)
        inputs[labels] = inputs[label]

        return inputs
    
    dataset_dict.map(tokenize,batch = True,remove_colums = ['text_a','label'])
        print(dataset_dict['train'])
    # 保存数据集
    dataset_dict.save_to_disk(config.PROCESSED_DATA_DIR)


if __name__ == '__main__':
    process()
