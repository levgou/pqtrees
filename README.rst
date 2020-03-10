.. role:: bash(code)
   :language: bash

.. role::py(code)
   :language: python3

PQTree
##############

How to install
------------------
1. Use python version >= 3.8
2. Clone this repo
3. Init a virtual env: :bash:`python3 -m venv /path/to/project`
4. :bash:`cd venv/bin ; source activate`
5. :bash:`pip install -r requirements.txt`
6. If using PyCharm set the project interpreter to be :bash:`/path/to/project/venv/bin/python3`
7. Installation of :code:`graphviz` might be required in order to visualize the PQTree

Usage
--------

Basic
^^^^^^^^
* Can be viewed in tests.py: compare_oren()

1. To construct a PQTree from permutations:
   :py:`PQTreeBuilder.from_perms(ListOfTuples)`

2. To visualize the tree with matplotlib:
   :py:`PQTreeVisualizer.show(pqtree)`

3. To compute the parenthesis representation:
   :py:`pqtree.to_parens()`

4. To convert to Json:
   :py:`pqtree.to_json(pretty=False | True)`

5. To calculate approximate frontier size:
   :py:`pqtree.approx_frontier_size()`

6. In order to calculate all the tree for permutations with duplications:
   :py:`PQTreeDup.from_perms(ListOfTuples)`

    * The above method will return an iterator,
      you can sort the trees by their aprox. frontier size
