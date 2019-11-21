.. include:: ./definitions.rst

****************************************
Common Intervals - Article
****************************************

Notation
==========================

|sigA|, |sigB|
----------------
The permutations we try to find common intervals for

* :math:`\sigma_A(i) = \alpha` denotes that |alpha| is the character at index |i|
* :math:`\sigma_A^{-1}(\alpha) = i` denotes that the index of |alpha| in |sigA| is |i|

|Pi_AB|
--------------

* :math:`\Pi_AB(i) = \sigma_B^{-1}(\sigma_A(i))`
    * Meaning - :math:`\Pi_AB(i) = j` the :math:`ith` element of |A| shows up at index |j| in |B|

Functions used in common intervals for |sigA| and |sigB|
-----------------------------------------------------------------------------

* :math:`l(x, y) = min(\Pi_AB(i) : i \in [x, y])`
    * The minimal / most left index of the the items in B that are at A at indices [x, y]

* :math:`u(x, y) = max(\Pi_AB(i) : i \in [x, y])`
    * The maximal / most right index of the the items in B that are at A at indices [x, y]

* :math:`f(x, y) = u(x, y) - l(x, y) - (y - x)`
    * :math:`f(x,y)` is the number of elements in
      :math:`\{ \sigma_B(i) | i \in [l(x,y), u(x,y)] \} - \{ \sigma_A(i) | i \in [x,y] \}`

    * (**Most right** - **most left**) - is the size of the interval in B, that contains all the elements from
      :math:`\sigma_A[x, y]`

    * Thus, |f_x_y| is the size of interval mentioned above,
        minus the same interval in A (which is just :math:`(x - y)`)

Wasteful |y|
------------------
* For a fixed |x| we call a |y| **wasteful** if :math:`f(x', y) > 0 : x' \leq x`
* Meaning that nor :math:`[x,y]` or any interval extended to the left is of the same **"size"** in A and B


Identifying wasteful |y|
-----------------------------

1. For some :math:`x>1 \; y>x` - if

  * :math:`u(x,y) < u(x, y')`
  * but :math:`u(x-1,y) = u(x-1, y')`  for some  :math:`y' > y`
  * Then :math:`f(x' y) > 0 for all x'< x`

2. For some :math:`x>1 y>x` - **If**

  * :math:`f(x,y) > f(x,y')` for some :math:`y' > y`
  * **Then** :math:`f(x' y) > 0` for all :math:`x' \leq x`

3 proposed algorithms:
============================

1. **LHP** -  simple |On2| time algorithm, whose expected running time becomes: |On| for two randomly generated permutations
2. **MNG** - A practically fast |On2| time algorithm using the reverse Monge property
3. **RC** - an :math:`O(n + K)` time algorithm, where :math:`K \leq \binom{n}{2}` is the number of common interval
