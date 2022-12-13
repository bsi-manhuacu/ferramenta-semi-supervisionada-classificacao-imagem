from os import listdir
from PIL import Image
from PIL import ImageFilter
from os import mkdir

import random
import shutil

import numpy as np

def read_dataset(dir_name):
    dataset = get_dataset(dir_name)
    dataset = sobel_filter(dataset)
    dataset = resize_imgs(dataset)
    dataset = crop_imgs(dataset)
    return dataset

def get_crossfolder_dataset(dir_name, k):
    dataset = [] # Lista que será o conjunto de dados final.
    img_id = 0   # Para configurar um valor de id para cada imagem.
    for i in range(10):
        if((i+1) != k):
            dir_f = dir_name+"/"+str(i+1)
            label = 0
        
            labels_folders = listdir(dir_f) # Pega o nome de todos os arquivos de imagens dentro do diretorio dir.
            for j in labels_folders:
                imgs_names = listdir(dir_f+"/"+j)

                for l in imgs_names: # Para cada imagem
                            
                    dataset.append({
                        "id": img_id,
                        "dir" : dir_f+"/"+l,
                        "img_rgb" : Image.open(dir_f+"/"+j+"/"+l).convert('RGB'), # Armazena os dados da imagem, como um objeto PIL
                        "label": label

                    })

                    img_id = img_id + 1

                label = label + 1

    random.shuffle(dataset)
    return dataset

# Função para retornar as imagens em um diretorio como um conjunto de dados
# O Conjunto de Dados é uma lista e cada elementos da lista é 
def get_dataset(dir_name) :
    
    dataset = [] # Lista que será o conjunto de dados final.
    img_id = 0   # Para configurar um valor de id para cada imagem.
    label = 0

    labels_folders = listdir(dir_name) # Pega o nome de todos os arquivos de imagens dentro do diretorio dir.
    for j in labels_folders:
        imgs_names = listdir(dir_name+"/"+j)

        for i in imgs_names: # Para cada imagem
                    
            dataset.append({
                "id": img_id,
                "dir" : dir_name+"/"+i,
                "img_rgb" : Image.open(dir_name+"/"+j+"/"+i).convert('RGB'), # Armazena os dados da imagem, como um objeto PIL
                "label": label

            })

            img_id = img_id + 1

        label = label + 1

    random.shuffle(dataset)
    return dataset

# Essa função aplica o filtro sobel = filtro para remover as cores e destacar o contraste local
def sobel_filter(dataset):
    for i in range(len(dataset)):
        #dataset[i]["img_sobel"] = dataset[i]["img_rgb"].filter(ImageFilter.FIND_EDGES)
        dataset[i]["img_sobel"] = dataset[i]["img_rgb"]
    return dataset

# Função para redimensionar a imagem. Por padrão a imagem é redimensionada para 256x256
def resize_imgs(dataset, dim=(256,256)):
    for i in range(len(dataset)):
        dataset[i]["img_resized"] = dataset[i]["img_sobel"].resize(dim)
    return dataset

# Função para recortar a imagem. Por padrão a imagem é recortada em relação ao centro (16, 16, 240, 240)
# Uma imagem de 256x256 passa a ter dimensões de 224x224, tamanho padrã para uma AlexNet
def crop_imgs(dataset, dim=(16, 16, 240, 240)):
    for i in range(len(dataset)):
        dataset[i]["img_cropped"] = dataset[i]["img_resized"].crop(dim)
    return dataset

def get_dataset_for_deepcluster(dataset):
    for i in range(len(dataset)):
        dataset[i]["img_cropped_npa"] = np.asarray(dataset[i]["img_cropped"])

    x = [] # x representa o conjunto de dados a ser agrupado. x é o conjunto imagens como vetores numpy
    for i in range(len(dataset)):
        x.append(dataset[i]["img_cropped_npa"])
        
    x = np.array(x)
    return x

def save_clustered_results(dataset, clusters, num_clusters, root_dir):

    # Diretorios de saídas:
    # Cada grupo será salvo em um diretório diferente.
    out_dirs = []

    for i in range(num_clusters):
        out_dirs.append(root_dir+"/c"+str(i)+"/")

    for i in range(num_clusters):
        try:
            shutil.rmtree(out_dirs[i]) # Tenta excluir o diretório de saida se ele existir
        except Exception as e: 
            print(e)
        try: 
            mkdir(out_dirs[i]) # Tenta criar o diretório de saída se ele não existir
        except Exception as e: 
            print(e)
    for i in range(len(dataset)):
        #idx = clusters[i].label
        dataset[clusters[i].id]["img_rgb"].save(out_dirs[clusters[i].major_label]+str(clusters[i].id)+".jpg")

def save_inference_results(dataset, clusters, num_clusters, root_dir):

    # Diretorios de saídas:
    # Cada grupo será salvo em um diretório diferente.
    out_dirs = []
    for i in range(num_clusters):
        out_dirs.append(root_dir+"/c"+str(i)+"/")

    for i in range(num_clusters):
        try:
            shutil.rmtree(out_dirs[i]) # Tenta excluir o diretório de saida se ele existir
        except Exception as e: 
            print(e)
        try: 
            mkdir(out_dirs[i]) # Tenta criar o diretório de saída se ele não existir
        except Exception as e: 
            print(e)
    for i in range(len(dataset)):
        idx = clusters[i]
        dataset[i]["img_rgb"].save(out_dirs[idx]+str(i)+".jpg")