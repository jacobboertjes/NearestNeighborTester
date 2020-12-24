from tester import LinearSearchTester, KDTreeTester, BallTreeTester, AnnoyTester, LSHTester
from generator import DataGenerator
from numpy.linalg import norm
import numpy as np
import sys


dg = DataGenerator()
d = 10

# Loop from 1-20 for 2^e total elements in the set
for e in range(10,21):
    n = 2**e
    q = 200


    data = {}
    data = dg.generate(n-q,d)

    queries = {}
    queries = dg.generate(q,d)

    testers = [LinearSearchTester(), KDTreeTester(), BallTreeTester(), LSHTester(), AnnoyTester()]

    print('\n')
    print('-------------------------')
    print('Size of set: {}'.format(n))
    print('-------------------------')
    print("==== Construction Tests ====")
    for tester in testers:
        print('Constructing {}'.format(tester.type))
        tester.setup(data)
    print("============================\n")

    print("======= Query Tests ========")
    print('Testing {}'.format(testers[0].type))
    expected = testers[0].test(queries)
    for tester in testers[1:]:
        print('Testing {}'.format(tester.type))
        neighbors = tester.test(queries)

        ## Get average percent error
        ## Special case for LSH, returns vectors
        sum = 0
        for i in range(len(queries)):
            if tester.type == "LSH":
                nn = neighbors[i]
            else:
                nn = data[neighbors[i]]
            opt = norm(queries[i]-expected[i])
            act = norm(queries[i]-nn)
            sum += abs(act - opt)/opt
            ##print('diff: {}'.format(opt-act))
        print('Average % error: {}'.format(sum/q*100))
    print("============================\n")
