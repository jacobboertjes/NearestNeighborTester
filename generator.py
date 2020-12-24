from random import randint
import numpy as np

class DataGenerator():
    def generate(self, n, d, min=-10, max=10):
        #print('Generating {} random {}-dimesnional points with values in the range [{}, {}]'.format(n, d, self.min, self.max))
        data = []
        for i in range(n):
            point = []
            for j in range(d):
                val = randint(min,max)

                #val = randint(self.min, self.max)

                point.append(val)
            data.append(np.array(point))
        return np.array(data)

if __name__ == '__main__':
    dg = DataGenerator(3, 20)
    dg.generate()
