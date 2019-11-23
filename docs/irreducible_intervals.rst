.. include:: ./definitions.rst

****************************************
Irreducible Intervals - Article
****************************************

Key points
=================

definitions
----------------------

|pi_|
^^^^^^^^^^^^^^^^^^^^^^
* |pi_1-pi_n| be a family of |k| permutations of |1-n|
* w.l.o.g. we assume in the following always that :math:`\pi_1 = id_n = (1,2,...,n)`

|pi_i| ; |pi_x_y|
^^^^^^^^^^^^^^^^^^^^^^
* :math:`\pi(i) = \alpha` means that the i'th element of |pi| is |alpha|
* :math:`x, y \in N, x \leq y` - :math:`[x, y]` denotes the set  :math:`\{x, x + 1,...,y\} \subseteq N`
  :math:`\pi[x,y] = \{ \pi(i) | i \in [x,y] \}` is called an interval of |pi|


Functions used in common intervals for |pi1| and |pi2|
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :math:`l(x, y) = min(\pi_2[x, y])`
* :math:`u(x, y) = max(\pi_2[x, y])`
* :math:`f(x, y) = u(x, y) - l(x, y) - (y - x)`
    * :math:`f(x, y)` counts the number of elements in :math:`\pi_1[l(x, y), u(x, y)]` \ |pi2| [x, y],
    * An interval :math:`\pi_2[x, y]` is a common interval of :math:`\Pi iff f(x, y) = 0`


Wasteful right interval end
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For a fixed |x|, a right interval end :math:`y>x` is called wasteful if it satisfies
:math:`f(x', y) > 0` for all :math:`x' \leq x`

Common Interval
---------------------
* A k-tuple :math:`c = ([l_1, u_1],..., [l_k, u_k])` is called a common interval of |pi_| iff

.. math::
    \pi_1[l_1, u_1] = \pi2[l_2, u_2] = ... = \pi_k[l_k, u_k]


* This allows to identify a common interval c with the contained elements, i.e.
  |c| |eqq| :math:`\pi_j[l_j, u_j]`  for  :math:`1 \leq j \leq k`

* Since :math:`\pi_1 = id_n`, the above set equals the index set :math:`[l_1, u_1]`, and we will refer
  to this as the **standard notation** of **c**

* The set of all common intervals of |pi_| is denoted as |C_Pi|

Algorithms
===================

Algorithm 1 (Reduce Candidate, RC)
----------------------------------------

* **Input:** A family :math:`\Pi = (\pi_1=id_n, \pi2)` of two permutations of |1-n|.
* **Output:** |C_Pi| in standard notation.

* Y data structure:
    * ylist - indices of non-wasteful right interval end candidates
    * llist, ulist - implement the functions l and u in order to compute |f| efficiently

.. code-block:: python3

    Y.initialize()                          # ylist initialized to store [n]
                                            # llist and ulist initialized with [n, n]

    for x in reversed(range(1, n, 1)):      # x = n − 1,..., 1
        Y.update()                          # Alg. 2
        y = x

        while y := ylist.succ(y) and f(x, y) == 0:
            output.add([l(x, y), u(x, y)])


Algorithm 2 (Update of data structure Y - Algorithm 1)
------------------------------------------------------------

* Only treats the case where :math:`\pi_2(x) > \pi_2(x+1)`, the other case is symmetric

: prepend x at the head of ylist
2: prepend [x, x] at the head of llist
3: while (u∗ ← ulist.head) has a successor u and val(u) < π2(x) do
4: delete u∗ from ulist and the corresponding elements from ylist
5: end while
6: y∗ ← end(u∗)
7: if (˜y ← ylist.succ(y∗)) is defined then
8: while f(x, y∗) > f(x, y˜) do
9: delete y∗ from ylist
10: y∗ ← ylist.pred(˜y)
11: end while
12: end if
13: update the left and right end of u∗ ← [x, y∗]

.. code-block:: python3

    def alg2(ylist, ulist, llist, pi2, x, f):
        ylist.prepend(x)
        llist.prepend([x, x])

        while u_star := ulist.popleft():
            if ylist.succ(u_star) and u_star < pi2[x]:
                ulist.remove(u_star)
                ylist.remove(u_star[0])
                ylist.remove(u_star[1])

        y_star = u_star[1]
        if y_tilde := ylist.succ(y_star):
            while f(x, y_star) > f(x, y_tilde):
                ylist.remove(y_star)
                y_star = ylist.pred(y_tilde)
