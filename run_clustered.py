from tester import LinearSearchTester, KDTreeTester, BallTreeTester, AnnoyTester, LSHTester
from generator import DataGenerator
from numpy.linalg import norm
import numpy as np
import sys


dg = DataGenerator()


# Loop over each dimension size
for d in [2,4,8,12,32,64,128,256,512,1024]:
    cdata = np.loadtxt('data_dim_txt/dim{}.txt'.format(d), delimiter=',', dtype=int)
    np.random.shuffle(cdata)

    n = 1024
    q = n//5

    min = 0
    if 2 <= d and d <= 12:
        ## When 2<= d <= 12, the values are all within 0-1000000
        max = 1000000
    else:
        ## Otherwise, the values are within 0-256
        max = 256

    data = {}
    data['uniform'] = dg.generate(n,d,min=min,max=max)
    data['clustered'] = cdata[:n]

    queries = {}
    queries['uniform'] = dg.generate(q,d,min=min,max=max)
    queries['clustered'] = dg.generate(q,d,min=min,max=max)

    testers = [LinearSearchTester(), KDTreeTester(), BallTreeTester(), LSHTester(), AnnoyTester()]

    for dtype in data:
        print('\n')
        print('-------------------------')
        print('{}D Data Distribution: {}'.format(d,dtype))
        print('-------------------------')
        print("==== Construction Tests ====")
        for tester in testers:
            print('Constructing {}'.format(tester.type))
            tester.setup(data[dtype])
        print("============================\n")

        print("======= Query Tests ========")
        print('Testing {}'.format(testers[0].type))
        expected = testers[0].test(queries[dtype])
        for tester in testers[1:]:
            print('Testing {}'.format(tester.type))
            neighbors = tester.test(queries[dtype])

            ## Get average percent error
            ## Special case for LSH, returns vectors
            sum = 0
            for i in range(len(queries[dtype])):
                if tester.type == "LSH":
                    nn = neighbors[i]
                else:
                    nn = data[dtype][neighbors[i]]
                opt = norm(queries[dtype][i]-expected[i])
                act = norm(queries[dtype][i]-nn)
                sum += abs(act - opt)/opt
                ##print('diff: {}'.format(opt-act))
            print('Average % error: {}'.format(sum/q*100))
        print("============================\n")
