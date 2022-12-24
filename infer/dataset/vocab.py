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
