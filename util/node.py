import numpy as np

class Node:

    def __init__(self, dist, left, right, id, num_labels):
        self.dist = dist
        self.left = left
        self.right = right
        if(left != None and right != None):
            self.left.set_parent(self)
            self.right.set_parent(self)
            self.objects_count = self.left.objects_count + self.right.objects_count
        else:
            self.objects_count = 1
        self.major_label = -1
        self.id = id
        self.confidence = np.full((num_labels,2),[0,1])
        self.labels_proportions = np.zeros(num_labels)
        self.error = self.objects_count
        self.split = False


    def set_parent(self, parent):
        self.parent = parent


    def get_major_label(self):
        return self.major_label

    def set_objects_count(self, num):
        self.objects_count = num

    def get_objects_count(self):
        return self.objects_count
        