import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from util.dataset import read_dataset
from util.dataset import get_dataset_for_deepcluster
from util.dataset import save_clustered_results

from util.cluster import construct_hierarchy
from util.cluster import query
from util.cluster import update_label_confidence
from util.cluster import update_partition
from util.cluster import get_leaves
from util.cluster import transductive_propagation

from util.cnn import get_alexnet
from util.cnn import remove_last_layer

from scipy import cluster

dir_name = input('forneça o diretório: ')
dir_results = input('forneça o diretório onde irá salvar os resultados: ')
num_class = int(input('forneça o número de classes: '))

max_deepcluster_it = 100

print('Carregando imagens...')
dataset = read_dataset(dir_name)
x = get_dataset_for_deepcluster(dataset)

print('Construindo modelo de Rede Neural Convolutiva...')
alexnet = get_alexnet(x, num_class)

print("Realizadno agrupamento profundo...")
for i in range(max_deepcluster_it):
    
    alexnet_wll = remove_last_layer(alexnet)        
    x_wards = alexnet_wll.predict(x)

    wards_dendrogram = cluster.hierarchy.ward(x_wards)
    clusters = cluster.hierarchy.cut_tree(wards_dendrogram, n_clusters=num_class)

    alexnet.fit(x, clusters, epochs=1)
    acc_local = alexnet.history.history['accuracy'][0]

    if (acc_local >= 0.9):
        break

node = construct_hierarchy(wards_dendrogram, num_class)
partition = [node]
queried_objs = []
to_query = 1

while (to_query):
    
    q_object = query(partition)
    
    img_show = dataset[q_object.id]["img_rgb"].show()
    q_label = int(input('Forneça o rótulo de classe da imagem: '))

    queried_objs.append(dataset[q_object.id]) 
    update_label_confidence(q_object, q_label, num_class)
    partition = update_partition(partition)

    objects = []
    get_leaves(node, objects)
    objects_labeled = transductive_propagation(objects)

    save_clustered_results(dataset, objects_labeled, num_class, dir_results)

    to_query = int(input('Deseja continuar o processo de rotulação (1) SIM - (0) NÃO: '))