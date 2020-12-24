# About
This repository contains an experimental tester for the following Nearest Neighbor Search algorithms:
1. Brute Force Search
2. k-d Tree
3. Ball Tree
4. Locality Sensitive Hashing
5. Annoy


The algorithms can be tested in two ways.
The first uses the low and high DIM datasets from http://cs.joensuu.fi/sipu/datasets/, along with a uniformly random dataset of matching size. 204 queries are run on each algorithm.
The second uses a low dimensional (10D) uniformly random dataset, for set sizes from 1024 to 1,048,576, doubling each increment, and running 200 queries each time.

All uniformly random data is generated automatically before test execution.


# Usage
### Clustered vs. Uniformly Random
`python3.7 run_clustered.py`
### Large Uniformly Random Set
`python3.7 run_large_set.py`
