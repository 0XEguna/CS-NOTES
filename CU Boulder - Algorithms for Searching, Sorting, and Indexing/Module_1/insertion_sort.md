## Wednesday, September 25th 2024

### Introduction: Insertion Sort

#### What is sorting?

* The process of ordering.

#### What is "Total Order" (Mathematics)?

Mathematically speaking,  total ordering is a relation  <= among the elements in a set with the following properties:

 * It is "reflexive" : i.e, a <= a for all elements a.

* It is "transitive" : i.e, if a <= b and b <= c then a <= c

* It is "anti-symmetric" : i.e, if a <= b and b <= a then a = b

* It is "total" : for any two elements a, b it is the case that a <= b or b <= a.

#### Two requirements for sorting:

1)  The Output must have the same number of elements as in the Input. Every element in the Output is in the Input. We are only allowed to permute the array.

2) The output must be sorted according to the provided order.

#### Insertion Sort: Pseudocode

Beware, the indexes in pseudocode start from 1 and are both inclusive unlike python.


* __Step 1 −__ If it is the first element, it is already sorted. return 1;

* __Step 2 −__ Pick next element

* __Step 3 −__ Compare with all elements in the sorted sub-list

* __Step 4 −__ Shift all the elements in the sorted sub-list that is greater than the value to be sorted

* __Step 5 −__ Insert the value

* __Step 6 −__ Repeat until list is sorted

```

Algorithm: Insertion-Sort(A) /*Being A an array*/
for j = 2 to A.length
   key = A[j]
   i = j – 1
   while i > 0 and A[i] > key
      A[i + 1] = A[i]
      i = i -1
   A[i + 1] = key

```

##### Computational Complexity

* __What is Computational Complexity?__

It is a fancy term to indicate how much does it cost (being the cost the number of resource consumed <Time, Space>).

Run time of this algorithm is very much dependent on the given input.

If the given numbers are sorted, this algorithm runs in O(n) time. If the given numbers are in reverse order, the algorithm runs in O(n2) time.