import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from transformers import CLIPProcessor, CLIPModel

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# in case of warning: To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false) The current process just got forked. Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collate_features(batch):
	return [[item[0] for item in batch], [item[1] for item in batch]]


#from huggingface_hub import snapshot_download
#snapshot_download(repo_id="plip")
class PLIPImageDataset(Dataset):
    def __init__(self, list_of_images, preprocessing, root_path=None):
        self.images = list_of_images
        self.preprocessing = preprocessing
        self.root_path = root_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.root_path is not None:
            name = os.path.join(self.root_path, self.images[idx])
        else:
            name = self.images[idx]

        images = self.preprocessing(Image.open(name).convert('RGB'))  # preprocess from clip.load
        return images, name



class PLIP_ZeroShot(object):
    def __init__(self, model_path=None, types_text=None, return_types_flag=None, device=None):
        assert model_path is not None

        self.device = device
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        print(f"PLIP model from path {model_path} has been loaded.")

        if types_text is None:
            self.types_text = ["tumor", "adipose", "stroma", "immune infiltrates lymphocytes", 
                          "gland", "necrosis or hemorrhage", "background or black", "non"]
        else:
            self.types_text = types_text

        if return_types_flag is None:
            self.return_types = None
        else:
            self.return_types = return_types_flag

        self.full_texts = ["a H&E image of {} tissue".format(label) for label in self.types_text]
        print(self.full_texts)
        

    def one_run(self, image=None):
        if type(image) is str: 
            image = Image.open(image)

        image = image.resize((224, 224))
        
        if self.device == torch.device('cuda'):
            image = torch.from_numpy(np.array(image))
            image = image.permute(2, 0, 1)
        
        inputs = self.processor(text=self.full_texts, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # softmax probs, 
        #print(probs)
        pred_cls_idx = probs.argmax(dim=1)

        if self.return_types is None:
            return pred_cls_idx
        else:
            return [self.types_text[each] in self.return_types for each in pred_cls_idx]

    def __call__(self, batch=None):
        # batch = [batch[idx] for idx in range(len(batch))]
        
        inputs = self.processor(text=self.full_texts, images=batch, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # softmax probs, 
        #print(probs)
        pred_cls_idx = probs.argmax(dim=1)
        # pred_type = [self.types_text[each] for each in pred_cls_idx]
        
        if self.return_types is None:
            return outputs.image_embeds, pred_cls_idx
        else:
            return outputs.image_embeds, [self.types_text[each] in self.return_types for each in pred_cls_idx]


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cpu'

    plip = PLIP_ZeroShot(model_path="/home/cyyan/Projects/HER2proj/scripts/plip/models/",
                         types_text=["tumor", "adipose", "stroma", "immune infiltrates lymphocytes", "gland", "necrosis or hemorrhage", "background or black", "non"],
                         return_types_flag=["tumor", "gland"],
                         device=device)
    
    data_path = "/home/cyyan/Projects/HER2proj/results/Yale_4Heatmaps/heatmap_production/HEATMAP_OUTPUT/sampled_patches/label_Unspecified_pred_0/topk_high_attention"
    imglist = os.listdir(data_path)

    if False:
        for imgname in imglist:
            name = os.path.join(data_path, imgname)
            
            image = Image.open(name)
            pred_type = plip.one_run(image)
            print(f">>>>>>>>>>>>>>>>>>>>>>>>{imgname}>>>>>>>>>>pred type: {pred_type}")
            # os.rename(name, os.path.join(data_path, pred_type+"+"+imgname))
    else:
        plip_trans = transforms.Compose([transforms.Resize(size=224)])
        dataset = PLIPImageDataset(imglist, plip_trans, root_path=data_path)

        kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
        loader = DataLoader(dataset=dataset, batch_size=16, **kwargs, collate_fn=collate_features)

        for batch, names in tqdm(loader, total=len(loader)):
            with torch.no_grad():	
                plip_feats, tissue_info = plip(batch)
                # print(tumorflag)
                for data in list(zip(names, tissue_info)):
                    print(data)
