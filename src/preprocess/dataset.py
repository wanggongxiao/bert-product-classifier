from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding , AutoTokenizer

from configuration import config


def get_dataset(dtype = 'train'):
    path = str(config.PROCESSED_DATA_DIR / dtype)
    return load_from_disk(path)

def get_dataloader(tokenizer ,dtype = 'train'):
    path = str(config.PROCESSED_DATA_DIR / dtype)
    dataset = load_from_disk(path)
    # dataset.set_format(type = 'torch')

    collate_fn = DataCollatorWithPadding(tokenizer,padding = True,return_tensors = 'pt')

    return DataLoader(dataset,
                      batch_size = 16,
                      shuffle = True,
                      collate_fn = collate_fn)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
    dataloader = get_dataloader(tokenizer,'train')
    for batch in dataloader:
        for k,v in batch.items():
            print(k,v)
        break
