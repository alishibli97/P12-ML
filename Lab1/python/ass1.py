import dtree
import monkdata as m
import drawtree_qt5 as d

entropy1 = dtree.entropy(m.monk1)
entropy2 = dtree.entropy(m.monk2)
entropy3 = dtree.entropy(m.monk3)

#print(entropy1)
#print(entropy2)
#print(entropy3)

num_attributes = len(m.attributes)

gain_monk1 = [ dtree.averageGain(m.monk1,m.attributes[i]) for i in range(num_attributes) ]
gain_monk2 = [ dtree.averageGain(m.monk2,m.attributes[i]) for i in range(num_attributes) ]
gain_monk3 = [ dtree.averageGain(m.monk3,m.attributes[i]) for i in range(num_attributes) ]

max1 = gain_monk1.index(max(gain_monk1))
max2 = gain_monk2.index(max(gain_monk2))
max3 = gain_monk3.index(max(gain_monk3))

#print(gain_monk1.index(max(gain_monk1)))
#print(gain_monk2.index(max(gain_monk2)))
#print(gain_monk3.index(max(gain_monk3)))

monk1_subsets = [ dtree.select(m.monk1,m.attributes[max1],i) for i in m.attributes[max1].values ]
gain_monk1_subsets = [ [  dtree.averageGain(monk1_subsets[i],m.attributes[j]) for j in range(num_attributes) ] for i in range(len(monk1_subsets)) ]

max_sub1 = [ gain_monk1_subsets[i].index(max(gain_monk1_subsets[i])) for i in range(len(monk1_subsets)) ]
#print(max_sub1)

most_commons = [ dtree.mostCommon(monk1_subsets[i]) for i in range(len(monk1_subsets)) ]
#print(most_commons)

#print(dtree.buildTree(m.monk1,m.attributes))

t1=dtree.buildTree(m.monk1,m.attributes)
t2=dtree.buildTree(m.monk2,m.attributes)
t3=dtree.buildTree(m.monk3,m.attributes)

#d.drawTree(t1)
#d.drawTree(t2)
#d.drawTree(t3)

# printing error on test sets
#print(1-dtree.check(t1, m.monk1test))
#print(1-dtree.check(t2, m.monk2test))
#print(1-dtree.check(t3, m.monk3test))

#print()

# printing error on train sets
#print(1-dtree.check(t1, m.monk1))
#print(1-dtree.check(t2, m.monk2))
#print(1-dtree.check(t3, m.monk3))

## results:
# monk3 has lowest error probably because of the 5% introduced noise, that led the model to generalize better to the test set
# we were expecting that the hierarchy of error would be:
# 1-Monk2: it needed 6 questions in the best and worst case
# 2-Monk3: it needed 4 questions in the worst case and 2 questions in the best case
# 3-Monk1: it needed 2 questions in the worst case and 1 question in the best case
# 
# Practically, Monk3 showed the least error, and we assume that is the case because of the introduced noise
# So as the hierarchy follows, Monk2 had the highest error, and then Monk1 followed.
#
# For the training and testing results:
# The training set will generally result in lower error since we used it to train the model initially, and the model already saw the data
# Vice versa, the test set is the new unseen d

## Starting part 6

import random
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


import matplotlib.pyplot as plt

list1 = [] # y-axis for monk1
list2 = [] # y-axis for monk2
list3 = [] # y-axis for monk3

fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # fractions on x-axis

for fraction in fractions:

    print("\nTesting split fraction =",fraction)

    # for monk1 dataset
    print("\nFor Monk1 dataset:")
    monk1train, monk1val = partition(m.monk1, fraction)

    current_tree = dtree.buildTree(monk1train,m.attributes)
    current_acc = dtree.check(current_tree,monk1val)
    print("Error before pruning is:",1-current_acc)

    list_trees = dtree.allPruned(current_tree)
    accuracies = [dtree.check(tree,monk1val) for tree in list_trees]
    best_acc = max(accuracies)
    best_tree = list_trees[accuracies.index(best_acc)]

    while(current_acc<best_acc):
        best_acc=current_acc

        list_trees = dtree.allPruned(current_tree)
        accuracies = [dtree.check(tree,monk1val) for tree in list_trees]
        current_acc = max(accuracies)
        best_tree = list_trees[accuracies.index(current_acc)]
        #print(best_acc)
    
    print("Error after pruning is:",1-current_acc)
    list1.append(1-current_acc)
    
    # for monk2 dataset
    print("\nFor Monk2 dataset:")
    monk2train, monk2val = partition(m.monk2, fraction)

    current_tree = dtree.buildTree(monk2train,m.attributes)
    current_acc = dtree.check(current_tree,monk2val)
    print("Error before pruning is:",1-current_acc)

    list_trees = dtree.allPruned(current_tree)
    accuracies = [dtree.check(tree,monk2val) for tree in list_trees]
    best_acc = max(accuracies)
    best_tree = list_trees[accuracies.index(best_acc)]

    while(current_acc<best_acc):
        best_acc=current_acc

        list_trees = dtree.allPruned(current_tree)
        accuracies = [dtree.check(tree,monk2val) for tree in list_trees]
        current_acc = max(accuracies)
        best_tree = list_trees[accuracies.index(current_acc)]
        #print(best_acc)

    print("Error after pruning is:",1-current_acc)
    list2.append(1-current_acc)
    
    # for monk3 dataset
    print("\nFor Monk3 dataset:")
    monk3train, monk3val = partition(m.monk3, fraction)

    current_tree = dtree.buildTree(monk3train,m.attributes)
    current_acc = dtree.check(current_tree,monk3val)
    print("Error before pruning is:",1-current_acc)

    list_trees = dtree.allPruned(current_tree)
    accuracies = [dtree.check(tree,monk3val) for tree in list_trees]
    best_acc = max(accuracies)
    best_tree = list_trees[accuracies.index(best_acc)]

    while(current_acc<best_acc):
        best_acc=current_acc

        list_trees = dtree.allPruned(current_tree)
        accuracies = [dtree.check(tree,monk3val) for tree in list_trees]
        current_acc = max(accuracies)
        best_tree = list_trees[accuracies.index(current_acc)]
        #print(best_acc)

    print("Error after pruning is:",1-current_acc)
    list3.append(1-current_acc)

plt.plot(fractions,list1,label='Monk1 dataset')
plt.plot(fractions,list2,label='Monk2 dataset')
plt.plot(fractions,list3,label='Monk3 dataset')
plt.legend()
plt.xlabel("Split fraction")
plt.ylabel("Classification error")
plt.show()