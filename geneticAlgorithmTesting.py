import pandas as pd
import numpy as np

from keras.models import model_from_json

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def main():
    """
    Tests the trained model
    """
    ##Reads in model
    jsonFile = open("model.json", "r")
    loadedModelJSON = jsonFile.read()
    jsonFile.close()

    ##Loads model and sets weights
    loadedModel = model_from_json(loadedModelJSON)
    loadedModel.load_weights("model.h5")
    print("Model Loaded")

    ##Loads in testing data
    testFeatures = np.load("TestingData.npy")
    testLabels = np.load("TestingLabels.npy")

    print(loadedModel.summary())

    ##Compiles model
    loadedModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    scores = loadedModel.evaluate(testFeatures, testLabels)

    ##Outputs statistics
    output = loadedModel.predict_classes(testFeatures)
    yActu = pd.Series(testLabels, name='Actual')
    yPred = pd.Series(np.ndarray.flatten(output), name='Predicted')
    dfConfusion = pd.crosstab(yActu, yPred)
    print(dfConfusion)

    ##Gets TN, TP, FN, FP using try catch blocks in the case that nothing is predicted for a class (ex. nothing predicted as 1).
    try:
        TN = dfConfusion[0][0]
    except:
        TN = 0

    try:
        TP = dfConfusion[1][1]
    except:
        TP = 0

    try:
        FN = dfConfusion[0][1]
    except:
        FN = 0

    try:
        FP = dfConfusion[1][0]
    except:
        FP = 0

    ##Counts number of positives and negatives
    numPositives = (np.count_nonzero(yActu))
    numNegatives = yActu.size - numPositives

    ##Gets precision, recall, specificity and fscore, using try, except blocks in case of a divide by 0 error.
    try:
        precision = TP / (TP + FP)
    except:
        precision = 0

    try:
        recall = TP / (TP + FN)
    except:
        recall = 0

    try:
        specificity = TN / (TN + FP)
    except:
        specificity = 0

    try:
        fScore = 2 * ((precision * recall) / (precision + recall))
    except:
        fScore = 0

    ##Displays metrics
    print('\n')
    print("METRICS")
    print("Accuracy: {}".format(scores[1]))
    print("Num Fraud: {}".format(numPositives))
    print("Precision: ", precision)
    print("Recall/Sensitivity: ", recall)
    print("Specificity: {}".format(specificity))
    print("F-score: {}".format(fScore))

    ##Calculates and displays ROC curve
    probs = loadedModel.predict_proba(testFeatures)
    auc = roc_auc_score(testLabels, probs)
    print("AUC: " + str(auc))
    fpr, tpr, thresholds = roc_curve(testLabels, probs)
    plt.plot(fpr, tpr, color = 'orange', label = "ROC")
    plt.plot([0,1],[0,1], color = 'darkblue', linestyle = '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()

main()