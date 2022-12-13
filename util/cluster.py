from scipy import cluster
from util.node import Node
import math
import random

def construct_hierarchy(wards_dendrogram, num_labels):
    tree = cluster.hierarchy.to_tree(wards_dendrogram)
    root = rec_tree(tree, num_labels)
    root.parent = None
    return root

def get_qq_leaves(root, queried):
    obj = []
    get_leaves(root, obj)
    qq = []
    for o in obj:
        for q in queried:
            if(o.id == q["id"]):
                qq.append(o)
    return qq

def get_leaves(root, objects):
    if(root.objects_count == 1):
        objects.append(root)
        return
    get_leaves(root.left, objects)
    get_leaves(root.right, objects)


def query(partition): 
    amm = 0
    for c in partition:
        amm = amm + c.error

    value = (random.randint(0,100)/100)*amm
    acc = 0

    ic_c = -1

    while(True):
        ic_c = ic_c+1
        acc = acc + partition[ic_c].error
        if(acc >= value and partition[ic_c].error != 0):
            break

    node = query_from_group(partition[ic_c])
    return node

def query_from_group(node):
    if(node.objects_count == 1):
        return node
    amm = node.left.error + node.right.error

    value = (random.randint(0,100)/100)*amm
    acc = 0

    acc = acc + node.left.error
    if(acc >= value and node.left.error != 0):
        return query_from_group(node.left)
    else:
        return query_from_group(node.right)
    

def rec_tree(wards_node, num_labels):
    if(wards_node.is_leaf()):
        node = Node(wards_node.dist, None, None, wards_node.id, num_labels)
        return node
    node_left = rec_tree(wards_node.get_left(), num_labels)
    node_right = rec_tree(wards_node.get_right(), num_labels)
    node = Node(wards_node.dist, node_left, node_right, wards_node.id, num_labels)
    return node

def update_label_proportions(node, label):
    while(node.parent != None):
        node.labels_proportions[label] = node.labels_proportions[label]+1
        node = node.parent

def update_label_confidence(node, label, num_labels):
    while(node != None):
        node.labels_proportions[label] = node.labels_proportions[label]+1
        for i in range(num_labels):
            frac = node.labels_proportions[i]/sum(node.labels_proportions)
            fs_corr = 1 - sum(node.labels_proportions)/node.objects_count
            #print('node = ',str(node.id),'sum(node.labels_proportions)', str(sum(node.labels_proportions)), ' node.objects_count = ', str(node.objects_count))
            delta = fs_corr/sum(node.labels_proportions)+math.sqrt(fs_corr*frac*(1-frac)/sum(node.labels_proportions))
            mean = frac*node.objects_count
            err = delta*node.objects_count
            node.confidence[i][0] = max(node.labels_proportions[i], mean - err)
            node.confidence[i][1] = min(node.objects_count - (sum(node.labels_proportions)-node.labels_proportions[i]), mean + err)
        
        major_label = -1
        max_count = 0
        for i in range(num_labels):
            is_representative = True
            for j in range(num_labels):
                if(i != j and node.confidence[i][0] <= 2*node.confidence[j][0]-node.objects_count):
                    is_representative = False
            if(is_representative and node.labels_proportions[i] > max_count):
                major_label = i
                max_count = node.labels_proportions[i]

        node.major_label = major_label

        if(node.major_label == -1):
            node.error = node.objects_count
        else:
            node.error = node.objects_count - node.confidence[node.major_label][0]

        if(node.objects_count > 1):
            split_error = node.left.error + node.right.error
            if(split_error < node.error and node.major_label != -1):
                node.error = split_error
                node.split = True

        node = node.parent

def transductive_propagation(nodes):
    nodes_labeled = []
    for i in nodes:
        if(i.major_label == -1):
            parent = i.parent
            while(parent.major_label == -1):
                parent = parent.parent 
        
            #i.major_label = parent.major_label
            node_labeled = Node(i.dist, None, None, i.id, len(i.labels_proportions))
            node_labeled.major_label = parent.major_label
            nodes_labeled.append(node_labeled)
        else:
            node_labeled = Node(i.dist, None, None, i.id, len(i.labels_proportions))
            node_labeled.major_label = i.major_label
            nodes_labeled.append(node_labeled)

    return nodes_labeled

def update_partition(nodes):
    partition = []

    while (len(nodes)>0):
        if(nodes[len(nodes)-1].split):
            node = nodes[len(nodes)-1]
            nodes.pop()
            nodes.append(node.left)
            nodes.append(node.right)
        else:
            partition.append(nodes.pop())
    
    return partition

def print_tree(node):
    if(node == None):
        return
    print('id = ',node.id, ' objects_count = ', node.objects_count, ' e=',node.error,' split = ',node.split, ' l* =', node.major_label, ' labeled objects = ', sum(node.labels_proportions))
    print_tree(node.left)
    print_tree(node.right)


def get_representative_images_2(wards_dendrogram, nk):
    tree = cluster.hierarchy.to_tree(wards_dendrogram)

    prune = [{
        "cluster":tree,
        "dist":tree.dist
    }]

    indice = 0
    while(len(prune) < nk):
        node = prune[indice]["cluster"]
        print(str(indice))
        if(node.is_leaf() != True):
            del prune[indice]
            prune.append({
                "cluster":node.get_left(),
                "dist":node.get_left().dist
            })
            prune.append({
                "cluster":node.get_right(),
                "dist":node.get_right().dist
            })
            prune.sort(key=lambda x: x.get('dist'), reverse=True)
        else:
            indice = indice + 1
            

    imgs = []

    parent = None
    for i in range(len(prune)):
        node = prune[i]["cluster"]
        while(node.get_count() > 5):
            parent = node
            if(node.get_left().get_count() > node.get_right().get_count()):
                node = node.get_left()
            else:
                node = node.get_right()
                
        imgs.append(node.pre_order(lambda x: x.id))

    return imgs


def get_representative_images(wards_dendrogram, nk):
    tree = cluster.hierarchy.to_tree(wards_dendrogram)

    prune = [{
        "cluster":tree,
        "dist":tree.dist
    }]

    indice = 0
    while(len(prune) < nk):
        node = prune[indice]["cluster"]
        print(str(indice))
        if(node.is_leaf() != True):
            del prune[indice]
            prune.append({
                "cluster":node.get_left(),
                "dist":node.get_left().dist
            })
            prune.append({
                "cluster":node.get_right(),
                "dist":node.get_right().dist
            })
            prune.sort(key=lambda x: x.get('dist'), reverse=True)
        else:
            indice = indice + 1
            

    imgs = []

    for i in range(len(prune)):
        node = prune[i]["cluster"]
        while(not node.is_leaf()):
            if(node.get_left().get_count() > node.get_right().get_count()):
                node = node.get_left()
            else:
                node = node.get_right()
                
        imgs.append(node.id)

    return imgs