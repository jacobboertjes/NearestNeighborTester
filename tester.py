from time import process_time_ns

from sklearn.neighbors import BallTree, KDTree
from lshashpy3 import LSHash
from annoy import AnnoyIndex
#from scipy.spatial import KDTree

from math import floor, log
import numpy as np
from fractions import Fraction


class Tester():
    def dist(self,a,b):
        return np.linalg.norm(a-b)

    def start_timer(self):
        self.start_time = process_time_ns()/1000000000

    def stop_timer(self):
        self.end_time = process_time_ns()/1000000000
        return Fraction(self.end_time - self.start_time)

    def test(self, queries):
        self.start_timer()
        out = self.query(queries)
        print('Test Execution Time: {}'.format(self.stop_timer()))
        return out

    def query(self, queries):
        raise Exception("Generic query unimplemented")

    def setup(self,data):
        self.start_timer()
        self.construct(data)
        print('Construction Execution time: {}'.format(self.stop_timer()))

    def type(self):
        return self.type


class LinearSearchTester(Tester):
    type = "Linear"
    def construct(self, data):
        self.data = data

    def query(self, queries):
        neighbors = []
        for q in queries:
            best = []
            best_dist = -1
            for p in self.data:
                if best_dist is -1 or self.dist(q,p)  < best_dist:
                    best = p
                    best_dist = self.dist(q,p)
            neighbors.append(best)
        return neighbors

class KDTreeTester(Tester):
    type = "KDTree"

    def construct(self, data):
        self.kdt = KDTree(data, metric="euclidean")
        #self.kdt = KDTree(data)

    def query(self, queries):
        return self.kdt.query(queries, k=1, return_distance=False)
        #d, i = self.kdt.query(queries, k=1, p=2)
        #return i

class BallTreeTester(Tester):
    type = "BallTree"
    def construct(self, data):
        self.bt = BallTree(data, metric="euclidean")

    def query(self, queries):
        return self.bt.query(queries, return_distance=False)

class AnnoyTester(Tester):
    type = "Annoy"
    def construct(self, data):
        ## Initialize using the number of dimensions in the first data item
        self.ai = AnnoyIndex(len(data[0]), "euclidean")
        for i in range(len(data)):
            self.ai.add_item(i, data[i])
        self.ai.build(5)

    def query(self, queries):
        neighbors = []
        for q in queries:
            neighbors.append(self.ai.get_nns_by_vector(q, 1, search_k=100, include_distances=False))
        return neighbors

class LSHTester(Tester):
    type = "LSH"
    def __init__(self):
        pass

    def construct(self,data):
        ##
        num_bits = floor(log(len(data)))
        d = len(data[0])
        self.lsh = LSHash(num_bits, d, num_hashtables=5)
        for v in data:
            self.lsh.index(v)

    def query(self, queries):
        neighbors = []
        for q in queries:
            nn = self.lsh.query(q, num_results=1, distance_func='euclidean')
            neighbors.append(np.asarray(nn[0][0][0]))
        return neighbors

if __name__ == "__main__":
    t = Tester()
    print("==== LINEAR SEARCH ====")
    l = LinearSearchTester()
    l.construct([[1,2],[3,4]])
    print(l.test([[0,0], [1,1], [3,5]]))
    print("=======================\n")
    print("=== KD TREE SEARCH ====")
    kd = KDTreeTester()
    kd.construct(np.array([[3,4],[1,2]]))
    print(kd.test(np.array([[0,0], [1,1], [3,5]])))
    print("=======================\n")
    print("=== BALL TREE SEARCH ===")
    bt = BallTreeTester()
    bt.construct(np.array([[3,4],[1,2]]))
    print(bt.test(np.array([[0,0], [1,1], [3,5]])))
    print("=======================\n")
