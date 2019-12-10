import pandas as pd
import numpy as np
import random


from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight ##Used if CSNN for fitness is used

##Reading in data and scaling it
df2 = pd.read_csv("creditcard.csv")
df = df2.sample(frac=1).reset_index(drop=True)

robustScaler = RobustScaler()

df['scaledAmount'] = robustScaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaledTime'] = robustScaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

#Scaled amount and time
scaledAmount = df['scaledAmount']
scaledTime = df['scaledTime']

#Insert into beginning of df
df.drop(['scaledAmount', 'scaledTime'], axis=1, inplace=True)
df.insert(0, 'scaledAmount', scaledAmount)
df.insert(1, 'scaledTime', scaledTime)


##Splitting up the data set
train, test = train_test_split(df, test_size=0.2)

##Diving up into train and test features and labels
trainFeatures = np.array(test.values[:,0:30])
trainLabels = np.array(test.values[:,-1])
testFeatures = np.array(test.values[:,0:30])
testLabels = np.array(test.values[:,-1])


class Net:

    def __init__(self, id):
        """
        Net object constructor, makes a new set of neural network weights, randomly assigned values of correct size
        Each Net object gets its own assigned weights and biases based on the genetic algorithm, which are tested against
        a multi-layered feed forward neural net using the training data set to determine the net's fitness.
        """
        self.id = id ##Used to optionally see which generation the net came from

        ##Initialise random weights and biases between -1 and 1
        self.weights1 = np.random.uniform(-1, 1, size=(30, 100))
        self.weights2 = np.random.uniform(-1, 1, size=(100, 100))
        self.weights3 = np.random.uniform(-1, 1, size=(100, 1))
        self.bias1 = np.random.uniform(-1, 1, size=(100))
        self.bias2 = np.random.uniform(-1, 1, size=(100))
        self.bias3 = np.random.uniform(-1, 1, size=(1))

        self.accuracy = 0
        self.fitness = None

    def reproduce(self, partner, weight1, weight2):
        """
        Produces a weighted average of neural net object and another parent, returning a new child
        """
        ##Creates a new child
        child = Net("temp")

        ##Takes weighted average of all weights and biases and assigns these values to the new child
        child.weights1 = np.add(self.weights1 * weight1, partner.weights1 * weight2)
        child.weights2 = np.add(self.weights2 * weight1, partner.weights2 * weight2)
        child.weights3 = np.add(self.weights3 * weight1, partner.weights3 * weight2)
        child.bias1 = np.add(self.bias1 * weight1, partner.bias1 * weight2)
        child.bias2 = np.add(self.bias2 * weight1, partner.bias2 * weight2)
        child.bias3 = np.add(self.bias3 * weight1, partner.bias3 * weight2)

        return child

    def mutate(self):
        """
        Mutate one of the children by randomizing weights and biases
        """
        self.weights1 = np.random.uniform(-1, 1, size=(30, 100))
        self.bias1 = np.random.uniform(-1, 1, size=(100))
        self.weights2 = np.random.uniform(-1, 1, size=(100, 100))
        self.bias2 = np.random.uniform(-1, 1, size=(100))
        self.weights3 = np.random.uniform(-1, 1, size=(100, 1))
        self.bias3 = np.random.uniform(-1, 1, size=(1))

        return self

    def testFitness(self):
        """
        Determines the accuracy of the specified weights of a Net object by setting the
        weights in a neural network and testing it on the testing data set
        """
        global trainFeatures
        global trainLabels

        ##Builds the neural network
        model = Sequential()
        model.add(Dense(units=100,
                        input_dim=30,
                        kernel_initializer='uniform',
                        activation='relu'))
        model.add(Dense(units=100,
                        kernel_initializer='uniform',
                        activation='relu'))
        model.add(Dense(units=1,
                        kernel_initializer='uniform',
                        activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        ##Gets the weights you want to manually assign to the net
        weights = [self.weights1, self.bias1, self.weights2, self.bias2, self.weights3, self.bias3]
        model.set_weights(weights)

        ##CSNN weighting updates - optionally included
        """
        classWeights = class_weight.compute_class_weight("balanced", np.unique(trainLabels), trainLabels)
        classWeights[1] *= 2
        """

        ##Evaluates the model on the testing data
        scores = model.evaluate(trainFeatures, trainLabels)

        ##Updates the individual nets accuracy and fitness attributes based on performance on training data
        self.accuracy = scores[1]

        ##Gets values for TN, TP, FN, FP, try catch blocks are used in the case that nothing is predicted as fraud
        output = model.predict_classes(testFeatures)
        yActu = pd.Series(testLabels, name='Actual')
        yPred = pd.Series(np.ndarray.flatten(output), name='Predicted')
        dfConfusion = pd.crosstab(yActu, yPred)

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

        """
        Optional other weight adjustment methods tested
        """
        ##self.fitness = FN ##Must sort in order of lowest to highest in the fitness function (reverse = False)
        ##self.fitness = scores[1] ##Must sort in order of highest to lowest in fitness functinon (reverse = True)
        ##self.fitness = FN / FN + TP ##Must sort in order of lowest to highest in fitness function (reverse = False)
        self.fitness = (TP + TN) - (10*FN) - FP ##Must sort in order of highest to lowest in fitness function (reverse = True)


        return self


def offspring(genNum, parents):
    """
    Creates all of the offspring of a given generation and gives them all new IDs
    """
    ##Determines the random weighting of each parent
    weight1 = round(random.uniform(0, 1), 3)
    weight2 = 1 - weight1

    ##Every parents reproduces with every other parent
    offspring = []
    random.shuffle(parents)
    for i in range(len(parents) - 1):
        x = i + 1
        for j in range(x, len(parents)):
            offspring.append(parents[i].reproduce(parents[j], weight1, weight2))


    ##Every offspring gets a label for their current generation and their number in the list
    for i in range(len(offspring)):
        offspring[i].id = "G" + str(genNum) + "N" + str(i)  ##G + generation its in + N for number + which number in list it is

    return offspring


def selection(offspring):
    """
    Ranks children in terms of fitness and picks the fittest 20
    """
    ##Picks 6 fittest
    newParents = []
    offspring.sort(key=lambda x: x.fitness, reverse=True)

    for i in range(0, 6):
        newParents.append(offspring[i])

    # Randomly mutate 2 solutions
    mutateNum = 0
    while mutateNum < 2:
        random.shuffle(offspring)
        if offspring[6] not in newParents:
            newParents.append(offspring[6].mutate())
            mutateNum += 1

    ##Randomly select 2 other solutions
    selectNum = 0
    while selectNum < 2:
        random.shuffle(offspring)
        if offspring[6] not in newParents:
            newParents.append(offspring[6])
            selectNum += 1

    ##Messed up - not enough children
    if len(newParents) != 10:
        print("Error - did not add enough offspring - offspring length: " + str(len(newParents)))

    return newParents


def geneticAlgorithm(epochs):
    """
    Evolves the weights for a given number of epochs, as long as a net with 100% accuracy hasn't been found
    """
    ##Makes original population of 10 neural nets
    currentGeneration = []
    for i in range(0, 10):
        id = "G" + str(0) + "N" + str(i)
        currentGeneration.append(Net(id))

    ##Runs for the number of epochs - add accuracy termination criteria
    e = 0
    previousAccuracy = 0
    differences = []

    ##While terminating criteria not met
    while e < epochs and previousAccuracy != 100.00:
        ##Reproduce current generation
        random.shuffle(currentGeneration)
        reproduced = offspring(e + 1, currentGeneration)

        ##Determine their fitness
        for net in reproduced:
            net.testFitness()

        ##Select best 10
        currentGeneration = selection(reproduced)

        ##Keeps track of the differences between generations
        diff = currentGeneration[0].accuracy - previousAccuracy
        differences.append(diff)

        previousAccuracy = currentGeneration[0].accuracy

        print("Epoch: " + str(e) + " Accuracy: " + str(currentGeneration[0].accuracy))

        e += 1


    """
    OPTIONAL - let's you view the differences for every epoch, used to determine the number of epochs 
    
    #for i in range(len(differences)):
        #print("Epoch: " + str(i) + str(differences[i]))
    """

    return currentGeneration[0]


def writeToJSON(net):
    global testFeatures
    global testLabels

    ##Builds the neural network
    model = Sequential()
    model.add(Dense(units=100,
                    input_dim=30,
                    kernel_initializer='uniform',
                    activation='relu'))
    model.add(Dense(units=100,
                    kernel_initializer='uniform',
                    activation='relu'))
    model.add(Dense(units=1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    ##Sets the weights to those of the chosen net
    weights = [net.weights1, net.bias1, net.weights2, net.bias2, net.weights3, net.bias3]
    model.set_weights(weights)

    ##Outputs model to a JSON
    modelJSON = model.to_json()

    with open("model.json", "w") as jsonFile:
        jsonFile.write(modelJSON)

    model.save_weights("model.h5")
    print("Saved Model")

    ##Saves the testing data
    np.save("TestingData", testFeatures)
    np.save("TestingLabels", testLabels)

    return


def main():
    """
    Trains network and writes it to the JSON
    """
    best = geneticAlgorithm(25)
    writeToJSON(best)


main()