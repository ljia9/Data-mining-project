import random
import numpy as np

def main():
    '''
    Generate random data naively by taking the average value of the attribute and then multiplying by a random value within the standard deviation
    '''
    print "Loading data..."
    raw_data = "datasets/z-training.txt"
    data = np.loadtxt(raw_data, delimiter=",")
    dataset = data[:,0:9]    

    mean = dataset.mean(axis=0)
    std = dataset.std(axis=0)
    #print mean

    # write to new file now
    f = open("new-data.txt", 'w')
    records = []

    with open("format/missing-values.txt") as d:
        for line in d:
            line = line.rstrip('\n')
            for i in range(0, len(mean)):
                a = 0
                coin = random.randint(1,2)
                if coin==1: a = -1
                else: a = 1

                val = mean[i] + (a * (random.uniform(0,1) * std[i]))
                if val < 0: val = 0
                line = line + ',' + str(val) 
            records.append(line)
    for rec in records:
        f.write(str(rec))
        f.write('\n')

    f.close() 
    #print "Check missing values now..."
    #check_missing()

def check_missing():
    '''
    For making sure submission file matches the right rows and columns
    '''
    raw_data = "format/dates.txt"
    dataset = []
    count = 0
    
    f = open("format/missing-values.txt", 'w')
    with open(raw_data) as a:
        for line in a:
            dataset.append(line)

    with open("sub.txt") as d:
        for line in d:
            if line not in dataset:
                l = line.rstrip('\n')
                f.write(str(l))
                f.write('\n')

    f.close()

if __name__ == "__main__":
    main()
