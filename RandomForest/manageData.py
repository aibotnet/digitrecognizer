
from sklearn import tree


def mergeOutput(file1, file2, output_file):
    file1.readline()
    file2.readline()

    count =0
    for val1 in file1:
        val2= file2.readline()
        if int(val1.split('\n')[0].split(',')[1]) == -1:
            output_file.write(val2)
            count+=1
        else:
            output_file.write(val1)


def main():
    file1 = open('rf_benchmark.csv')
    file2 = open('/home/vkthakur/PycharmProjects/my-practice-codes/DigitRecognizer/KNN/knn_benchmark.csv')
    out = open('rf_benchmark.csv','w')


if __name__=='__main__':
    main()