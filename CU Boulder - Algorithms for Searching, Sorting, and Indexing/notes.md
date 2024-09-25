# CU Boulder - Algorithms for Searching, Sorting, and Indexing

## Wednesday, September 18th 2024

### Course overview:

#### What is an algorithm?

* An Algorithm is a computational procedure (or recipe) for solving a problem we are interested in. They have inputs and they provide ouputs

What is an __Instance__ (of the problem)?

* Whenever we plug-in values for the input, we call it the instance of the problem.  

Once we have the instance of the model, the algorithm runs in a mechanical manner (there is no creativity, it is a step by step process.) The we obtain the result.

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

```
    for i = j-1 down to 1:  

        if array[i] > array[i+1]:

            swap(array[i], array[i+1])

        else:

            break