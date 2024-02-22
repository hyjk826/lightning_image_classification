import os

def label_encoder(data_path):
    label = []
    for i in os.listdir(data_path):
        label.append(i)
    return {key : idx for idx, key in enumerate(label)}

def label_decoder(data_path):
    label_encoder_dict = label_encoder(data_path)
    return {val : key for key, val in label_encoder_dict.items()}

def load_img_paths(data_path):
    
    img_paths = []
    labels = []
    
    for i in os.listdir(data_path):
        for j in os.listdir(os.path.join(data_path, i)):
            img_paths.append(os.path.join(data_path, i, j))        
            labels.append(i)

    return img_paths, labels