from ID3 import ID3
from utils import *
from sklearn.model_selection import KFold

"""
Make the imports of python packages needed
"""

"""
========================================================================
========================================================================
                              Experiments 
========================================================================
========================================================================
"""
target_attribute = 'diagnosis'


# ========================================================================
def basic_experiment(x_train, y_train, x_test, y_test, formatted_print=False):
    """
    Use ID3 model, to train on the training dataset and evaluating the accuracy in the test set.
    """

    # TODO:
    #  - Instate ID3 decision tree instance.
    #  - Fit the tree on the training data set.
    #  - Test the model on the test set (evaluate the accuracy) and print the result.

    acc = None

    # ====== YOUR CODE: ======
    ourTree = ID3(["B","M"])
    ourTree.fit(x_train, y_train)
    results = ourTree.predict(x_test)
    correct = 0
    for i in range(len(results)):
        if results[i] == y_test[i]:
            correct += 1
    acc = correct/(len(results))
    # ========================

    assert acc > 0.9, 'you should get an accuracy of at least 90% for the full ID3 decision tree'
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)


# ========================================================================

def best_m_test(x_train, y_train, x_test, y_test, min_for_pruning):
    """
        Test the pruning for the best M value we have got from the cross validation experiment.
        :param: best_m: the value of M with the highest mean accuracy across folds
        :return: acc: the accuracy value of ID3 decision tree instance that using the best_m as the pruning parameter.
    """

    # TODO:
    #  - Instate ID3 decision tree instance (using pre-training pruning condition).
    #  - Fit the tree on the training data set.
    #  - Test the model on the test set (evaluate the accuracy) and return the result.

    acc = None

    # ====== YOUR CODE: ======
    ourTree = ID3(["B","M"],min_for_pruning)
    ourTree.fit(x_train, y_train)
    results = ourTree.predict(x_test)
    correct = 0
    for i in range(len(results)):
        if results[i] == y_test[i]:
            correct += 1
    acc = correct/(len(results))
    # ========================

    return acc

def cross_validation_experiment(x_train, y_train, x_test, y_test,):

    mArr = [20,50,100,150,300]
    results = []
    kf = KFold(shuffle=True, n_splits=5, random_state=208548834)
    kf.get_n_splits(x_train)
    for m in mArr:
        mAcc = 0
        maxLen = 0
        for i, (trainIndex, testIndex) in enumerate(kf.split(x_train)):
            trainingSetY = []
            trainingSetX = []
            testSetY = []
            testSetX = []
            for x in trainIndex:
                trainingSetX.append(x_train[x])
                trainingSetY.append(y_train[x])
            for x in testIndex:
                testSetX.append(x_train[x])
                testSetY.append(y_train[x])
            mAcc += best_m_test(trainingSetX, trainingSetY, testSetX, testSetY, m)
            if i > maxLen:
                maxLen = i
        mAcc = mAcc / 5
        results.append(mAcc)
    for i in range(len(results)):
        results[i] = results[i] * 100
    util_plot_graph(mArr, results, "M Values", "Accuracy (%)")
    maxIdx = np.argmax(results)
    return mArr[maxIdx]


# ========================================================================
if __name__ == '__main__':
    attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    data_split = get_dataset_split(train_dataset, test_dataset, target_attribute)

    """
    Usages helper:
    (*) To get the results in ???informal??? or nicely printable string representation of an object
        modify the call "utils.set_formatted_values(value=False)" from False to True and run it
    """
    formatted_print = True
    basic_experiment(*data_split, formatted_print)

    """
       cross validation experiment
       (*) To run the cross validation experiment over the  M pruning hyper-parameter 
           uncomment below code and run it
           modify the value from False to True to plot the experiment result
    """
    plot_graphs = True
    #best_m = cross_validation_experiment(*data_split)
    best_m = 20
    print(f'best_m = {best_m}')

    """
        pruning experiment, run with the best parameter
        (*) To run the experiment uncomment below code and run it
    """
    acc = best_m_test(*data_split, min_for_pruning=best_m)
    assert acc > 0.95, 'you should get an accuracy of at least 95% for the pruned ID3 decision tree'
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)
