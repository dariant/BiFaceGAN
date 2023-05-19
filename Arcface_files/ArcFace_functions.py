import numpy as np 
import torch 
from Arcface_files.backbones import get_model

#########################################
""" Replaced with torch cosine similarity """
# def cosine_similarity(x, y):
#     dot = np.sum(np.multiply(x, y), axis=1)
#     norm = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
#     similarity = np.clip(dot/norm, -1., 1.)
#     return similarity

#########################################
def preprocess_image_for_ArcFace(img): 
    #img = cv2.resize(img, (112, 112))
    img = img.resize((112, 112))
    img = np.array(img)
    # print(np.array(img).shape)
    img = np.transpose(img, (2, 0, 1))
    
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    
    return img 

#########################################
def prepare_locked_ArcFace_model():
    arcface_model = get_model("r100", fp16=True) # TODO turned to true
    arcface_weights = "Arcface_files/ArcFace_r100_ms1mv3_backbone.pth"
    arcface_model.load_state_dict(torch.load(arcface_weights))
    arcface_model.eval()
    for param in arcface_model.parameters():
        param.requires_grad = False
    
    #print(arcface_model)
    return arcface_model