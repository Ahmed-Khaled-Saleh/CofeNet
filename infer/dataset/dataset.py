import torch
from tqdm import tqdm

from torch.utils.data import Dataset
from infer.utils import utils
from infer.dataset import vocab

class ExpDatasetBase(Dataset):
    LBID_IGN = -1

    def __init__(self, file_path, device=None):
        self.device = torch.device('cpu') if device is None else device

        self.map_tg2tgid = {tag: idx for idx, tag in enumerate(utils.load_text_file_by_line(file_path))}
        self.map_tgid2tg = {idx: tag for tag, idx in self.map_tg2tgid.items()}

        self.org_data = load_data_from_file(file_path)

class DatasetBert(ExpDatasetBase):
    def __init__(self, file_path, device=None):
        super().__init__(file_path, device)

        self.vocab = vocab.VocabularyBert.load_vocabulary()
        self.tkidss, self.wdlenss, self.lbidss, self.tk_lengths, self.wd_lengths = [], [], [], [], []
        for item in tqdm(self.org_data):
            tkids, wdlens, lbids = [self.vocab.ID_CLS], [], []

            for wd in item['tokens']:
                wd_tkids = self.vocab.wd2ids(wd)
                tkids.extend(wd_tkids)
                wdlens.append(len(wd_tkids))
                #lbids.append(self.map_tg2tgid[tg])

            self.tkidss.append(tkids)
            self.wdlenss.append(wdlens)
            #self.lbidss.append(lbids)

            self.tk_lengths.append(len(tkids))
            self.wd_lengths.append(len(wdlens))

    def __len__(self):
        return len(self.tkidss)

    def __getitem__(self, idx):
        return {
            'tkids': self.tkidss[idx],
            #'lbids': self.lbidss[idx],
            'wdlens': self.wdlenss[idx],
            'tk_length': self.tk_lengths[idx],
            'wd_length': self.wd_lengths[idx],
            #'lbstrs': self.org_data[idx]['labels']
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
            #lbidss.append(item['lbids'] + [self.LBID_IGN] * wd_num_pad)
            #lbstrss.append(item['lbstrs'])

        output = {
            'tkidss': torch.tensor(tkidss).to(self.device),
            'attention_mask': torch.tensor(attention_mask).to(self.device),
            'wdlens': torch.tensor(wdlens).to(self.device),
            'lengths': torch.tensor(wd_lengths).to(self.device)
        }

        #lbidss = torch.tensor(lbidss).to(self.device)
        return output
