from infer.dataset import loader
import torch
from torch.utils.data import SequentialSampler
from infer.utils import utils
from infer.dataset import dataset
from infer.model.mod_bert import  ModelBert_Cofe

def get_preds_trues(dataset, model_path = '/kaggle/working/CofeNet/model_6000.bin', batch_size= 32):
    dataloder = loader.SingleDataLoader(dataset=dataset, batch_size=batch_size,
                                sampler=SequentialSampler(dataset), collate_fn=dataset.collate)
    preds, labels = [], []
    model = ModelBert_Cofe()
    model = model.load_state_dict(torch.load(model_path, map_location='cpu'))

    for batch_data in dataloder:
        model.eval()
        with torch.no_grad():
            batch_preds = model.predict(batch_data)
            print(batch_preds)
            break
        #print(batch_data)



if __name__ == '__main__':


    infer_str = \
    ['Kyiv has strongly denied the accusations and said that Russia is using nuclear blackmail in order to try to block support \
        for its successful counteroffensive against the Russian invasion force.',\

    'The US president, Joe Biden, on Wednesday said that he had spent “a lot of time” discussing whether Russia may be preparing to use a \
        tactical nuclear weapon in Ukraine.', \
            
    '“We have never said anything about the possible use of nuclear weapons by Russia, but only hinted at the statements made by the leaders\
         of western countries,” Putin said in his remarks.'
    ]

    file_path = utils.read_write_str(infer_str, "infer_file.txt")
    #data = load_data_from_file(file_path)
    #print(data)
    #print(load_json_file_by_line(file_path))
    DataBert = dataset.DatasetBert(file_path)
    get_preds_trues(DataBert)


    #ModelBert_Cofe().load_state_dict(torch.load(model_path, map_location='cpu'))