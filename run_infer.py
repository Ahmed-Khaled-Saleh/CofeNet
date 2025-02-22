from infer.dataset import loader
import torch
from torch.utils.data import SequentialSampler
from infer.utils import utils
from infer.dataset import dataset
from infer.model.mod_bert import ModelBert_Cofe





def tgidss2tgstrss(tgidss, tags_file_path ,lengths=None):
        tgstrss = []
        map_tg2tgid = {tag: idx for idx, tag in enumerate(utils.load_text_file_by_line(tags_file_path))}
        map_tgid2tg = {idx: tag for tag, idx in map_tg2tgid.items()}
        
        
        if lengths is None:
            for tgids in tgidss:
                tgstrss.append([map_tgid2tg[tgid] for tgid in tgids])
        else:
            for tgids, length in zip(tgidss, lengths):
                tgstrss.append([map_tgid2tg[tgid] for tgid in tgids[:length]])
        return tgstrss




def get_preds_trues(dataset, file_path, model_path = '/kaggle/working/model_6000.bin', batch_size= 32):
    dataloder = loader.SingleDataLoader(dataset=dataset, batch_size=batch_size,
                                sampler=SequentialSampler(dataset), collate_fn=dataset.collate)
    preds = []
    model = ModelBert_Cofe()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    for batch_data in dataloder:
        model.eval()
        with torch.no_grad():
            batch_preds = model.predict(batch_data)
            
            
            batch_pred_strs = tgidss2tgstrss(
                batch_preds.data.cpu().numpy() if not isinstance(batch_preds, list) else batch_preds, file_path,
                batch_data['lengths'].cpu().numpy())

            preds.extend(batch_pred_strs)
    return preds
            


if __name__ == '__main__':


    infer_str = \
    ['Kyiv has strongly denied the accusations and said that Russia is using nuclear blackmail in order to try to block support \
        for its successful counteroffensive against the Russian invasion force.',

    'The US president, Joe Biden, on Wednesday said that he had spent “a lot of time” discussing whether Russia may be preparing to use a \
        tactical nuclear weapon in Ukraine.', 
            
    '“We have never said anything about the possible use of nuclear weapons by Russia, but only hinted at the statements made by the leaders\
         of western countries,” Putin said in his remarks.',
         
    'The US president, Joe Biden, on Wednesday said that he had spent “a lot of time” discussing whether Russia may be preparing to use a tactical nuclear weapon in Ukraine.“Let me just say Russia would be making an incredibly serious mistake if it were to use a tactical nuclear weapon,” he told reporters. “I’m not guaranteeing that it’s a false-flag operation yet. We don’t know. It would be a serious, serious mistake.”During his speech at the Valdai Club, a Kremlin-aligned foreign policy thinktank, on Thursday, Putin said that assertions about Russia’s possible use of nuclear weapons were meant to scare its supporters by indicating “what a bad country Russia is”. “We have never said anything about the possible use of nuclear weapons by Russia, but only hinted at the statements made by the leaders of western countries,” Putin said in his remarks.'
    ]

    file_path = utils.read_write_str(infer_str, "infer_file.txt")
    DataBert = dataset.DatasetBert(file_path)
    tag_file_path = '/kaggle/working//CofeNet/res/polnear/tag.txt'
    print(get_preds_trues(DataBert, tag_file_path))
