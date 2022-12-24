import json
import typing

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





from abc import ABCMeta
from abc import abstractmethod
from pytorch_transformers import BertTokenizer


class VocabularyBase(metaclass=ABCMeta):
    TK_PAD = '[PAD]'
    TK_UNK = '[UNK]'

    @abstractmethod
    def wd2ids(self, word):
        raise NotImplemented


class VocabularyBert(VocabularyBase):
    TK_CLS = '[CLS]'
    TK_MSK = '[MASK]'
    TK_SEP = '[SEP]'

    def __init__(self, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer
        self.ID_PAD, self.ID_UNK, self.ID_CLS, self.ID_MSK, self.ID_SEP = \
            self.tokenizer.convert_tokens_to_ids([self.TK_PAD, self.TK_UNK, self.TK_CLS, self.TK_MSK, self.TK_SEP])

    def wd2ids(self, word):
        if not word:
            ret = [self.ID_UNK]
        else:
            ret = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
            if not ret:
                ret = [self.ID_UNK]
        ret = [x if x not in [self.ID_PAD, self.ID_CLS, self.ID_MSK, self.ID_SEP] else self.ID_UNK for x in ret]
        return ret

    @classmethod
    def load_vocabulary(cls):

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #tokenizer.save_pretrained(exp_conf.model_vocab_dir)
        return cls(tokenizer)



import torch
from tqdm import tqdm

from torch.utils.data import Dataset


class ExpDatasetBase(Dataset):
    LBID_IGN = -1

    def __init__(self, file_path, device=None):
        self.device = torch.device('cpu') if device is None else device

        self.map_tg2tgid = {tag: idx for idx, tag in enumerate(load_text_file_by_line(file_path))}
        self.map_tgid2tg = {idx: tag for tag, idx in self.map_tg2tgid.items()}

        self.org_data = load_data_from_file(file_path)

class DatasetBert(ExpDatasetBase):
    def __init__(self, file_path, device=None):
        super().__init__(file_path, device)

        self.vocab = VocabularyBert.load_vocabulary()
        self.tkidss, self.wdlenss, self.lbidss, self.tk_lengths, self.wd_lengths = [], [], [], [], []
        for item in tqdm(self.org_data):
            tkids, wdlens, lbids = [self.vocab.ID_CLS], [], []

            for wd, tg in zip(item['tokens'], item['labels']):
                wd_tkids = self.vocab.wd2ids(wd)
                tkids.extend(wd_tkids)
                wdlens.append(len(wd_tkids))
                lbids.append(self.map_tg2tgid[tg])

            self.tkidss.append(tkids)
            self.wdlenss.append(wdlens)
            self.lbidss.append(lbids)

            self.tk_lengths.append(len(tkids))
            self.wd_lengths.append(len(wdlens))

    def __len__(self):
        return len(self.tkidss)

    def __getitem__(self, idx):
        return {
            'tkids': self.tkidss[idx],
            'lbids': self.lbidss[idx],
            'wdlens': self.wdlenss[idx],
            'tk_length': self.tk_lengths[idx],
            'wd_length': self.wd_lengths[idx],
            'lbstrs': self.org_data[idx]['labels']
        }

    def collate(self, batch):
        """
        And for DataLoader `collate_fn`.
        :param batch: list of {
                'tkids': [tkid, tkid, ...],
                'lbids': [lbid, lbid, ...],
                'wdlens': [wdlen, wdlen, ...],
                'tk_length': len('tkids'),
                'wd_length': len('lbids') or len('wdlens'),
                'lbstrs': len('lbids')
            }
        :return: (
                    {
                        'tkidss': tensor[batch, seq],
                        'attention_mask': tensor[batch, seq],
                        'wdlens': tensor[batch, seq],
                        'lengths': tensor[batch],
                    }
                    ,
                    lbidss: tensor[batch, seq]
                    ,
                    lbstrss: list[list[string]]
            )
        """
        tk_lengths = [item['tk_length'] for item in batch]
        wd_lengths = [item['wd_length'] for item in batch]
        tk_max_length = max(tk_lengths)
        wd_max_length = max(wd_lengths)

        tkidss, attention_mask, wdlens, lbidss, lbstrss = [], [], [], [], []
        for item in batch:
            tk_num_pad = tk_max_length - item['tk_length']
            wd_num_pad = wd_max_length - item['wd_length']

            tkidss.append(item['tkids'] + [self.vocab.ID_PAD] * tk_num_pad)
            attention_mask.append([1] * item['tk_length'] + [0] * tk_num_pad)
            wdlens.append(item['wdlens'] + [0] * wd_num_pad)
            lbidss.append(item['lbids'] + [self.LBID_IGN] * wd_num_pad)
            lbstrss.append(item['lbstrs'])

        output = {
            'tkidss': torch.tensor(tkidss).to(self.device),
            'attention_mask': torch.tensor(attention_mask).to(self.device),
            'wdlens': torch.tensor(wdlens).to(self.device),
            'lengths': torch.tensor(wd_lengths).to(self.device)
        }

        lbidss = torch.tensor(lbidss).to(self.device)
        return output, lbidss, lbstrss





if __name__ == '__main__':


    infer_str = \
    ['Kyiv has strongly denied the accusations and said that Russia is using nuclear blackmail in order to try to block support \
        for its successful counteroffensive against the Russian invasion force.',\

    'The US president, Joe Biden, on Wednesday said that he had spent “a lot of time” discussing whether Russia may be preparing to use a \
        tactical nuclear weapon in Ukraine.', \
            
    '“We have never said anything about the possible use of nuclear weapons by Russia, but only hinted at the statements made by the leaders\
         of western countries,” Putin said in his remarks.'
    ]

    file_path = read_write_str(infer_str, "infer_file.txt")
    #data = load_data_from_file(file_path)
    #print(data)
    #print(load_json_file_by_line(file_path))
    DataBert = DatasetBert(file_path)
