import torch
import pandas as pd
import sys 
from networks import *
from dataset import *

class_names = ['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']
       
##### 
#python src/<path-to-prediction-program> <input-data-csv-filename> <output-prediction-csv-path>
#####
assert(len(sys.argv) == 3)
input_pth = sys.argv[1]
output_pth = sys.argv[2]
model_pth = 'experiments/resnet18/model_best.pth.tar'
IS_CUDA = torch.cuda.is_available()


def load_checkpoint(path, model):
    if not IS_CUDA:
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
        

    return model
    
def eval():
    test_csv = pd.read_csv(input_pth).loc[:,'Study'] # take study only
    test_dataset = CheXpertDataset(test=input_pth)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle = False)
    model = load_checkpoint(model_pth, modified_resnet18())
    
    all_preds = None
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.float()
            input = input.float()
            if IS_CUDA:
                target = target.cuda()
                input = input.cuda()
            preds = model(input)
            preds = ((preds.sigmoid())>0.5)
            if type(all_preds) == type(None):
                all_preds = preds
            else:
                all_preds = torch.cat((all_preds, preds))
                
    assert(all_preds.shape[0] == test_csv.shape[0])
    # atelectasis - 8, cardiomegaly - 2, consolidation - 6, edema - 5, pleural effusion - 10
    df = pd.DataFrame({'Atelectasis': all_preds[:,8].squeeze().numpy(), 'Cardiomegaly': all_preds[:,2].squeeze().numpy(), 'Consolidation':all_preds[:,6].squeeze().numpy(), 'Edema':all_preds[:,5].squeeze().numpy(), 'Pleural Effusion':all_preds[:,10].squeeze().numpy()})
    
    assert(test_csv.shape[0] == df.shape[0])
    test_csv = pd.concat([test_csv, df], axis=1)
    print(test_csv.shape)
    test_csv.to_csv(output_pth, index=False)
    

if __name__ == '__main__':
    eval()
    
    