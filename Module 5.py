import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.utils import shuffle
import random

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100)

# Random.shuffle kan niet want anders zouden de lijsten niet aligned zijn met elkaar. Deze functie kan dat blijkbaar wel.
data_shuffled, targets_shuffled, images_shuffled = shuffle(digits.data, digits.target, digits.images) #Lijsten van dataset randomize. ( Maar wel de posities behouden tot elkaar ).

#Je kan een for loop maken die een derde van de lijst random uitkiest en verwijderd uit de originele lijst.
trainingData,trainingTarget = data_shuffled[:1198], targets_shuffled[:1198] #2/3 van de random data is training set.
testData = data_shuffled[1198:] #De rest ( 1/3 ) van de data is test set.
images = images_shuffled[1198:] #Het laatste deel ( 1/3 ) zijn de images van de test set om te laten zien.

clf.fit(trainingData,trainingTarget)

print(clf.predict(testData[-4:-3])) #Beetje flauw, maar hij wil perse 2d-array hebben ook als er maar 1 element inzit

#For loop die voor elk element in de testData en dan het resultaat van de predict aan results toevoegd.
results = []
for i in range(len(testData)):
    results.append(clf.predict(testData[i:i+1])) #Zeker flauw haha. :/

#For loop die door alle resultaten loopt en vergelijkt met de targets.
testTarget = targets_shuffled[1198:] #Alle targets die bij de test data horen.
correct = 0 #Correcte guesses die zijn gemaakt.
incorrect = 0 #Incorrecte guesses die zijn gemaakt.

print( len(results), len(testTarget)) #Check om te kijken of de lijsten even lang zijn.

for i in range(len(results)):
    if( results[i] == testTarget[i] ):
        correct += 1
    else:
        incorrect += 1

print( correct, incorrect )
print( "Correct: " + str( ( correct / 599 ) * 100 ) + "%")
print( "Incorrect: " + str( ( incorrect / 599 ) * 100 ) + "%" )

print( results[-4] ) #Check of het resultaat in results hetzelfde is als het vorige resultaat met dezelfde data.

plt.imshow(images[-4], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()