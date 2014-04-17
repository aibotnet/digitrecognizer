from numpy import *
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def img2vector(line):
    returnVect = zeros((1,784))
    i=0
    for num in line.split(','):
        if len(num)!=0:
            returnVect[0,i] = int(num)
            i=i+1
    return returnVect


def handwritingClassTest():
    hwLabels = []
    fr= open('train.csv')
    m=len(fr.readlines())-1
    trainingMat = zeros((m,784))
    fr.close()

    fr= open('train.csv')
    fr.readline()
    for i in range(m):
        line=fr.readline()
        hwLabels.append(int(line[0]))
        trainingMat[i,:] = img2vector(line[2:])

    tr = open('test.csv')
    mTest = len(tr.readlines())
    tr = open('test.csv')
    tr.readline()
    out=open('knn_benchmark_test.csv','a')
    out.write('ImageId,Label\n')
    for i in range(mTest-1):
        vectorUnderTest = img2vector(tr.readline())
        classifierResult = classify0(vectorUnderTest,trainingMat, hwLabels, 3)
        out.write('%d,%s\n'%(i+1,classifierResult))

        print 'Predicted Digit :  %s'%classifierResult

    out.close()

def main():
    handwritingClassTest()

if __name__=='__main__':
    main()
