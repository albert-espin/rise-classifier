# RISE Rule-Based Classifier

The Rule Induction from a Set of Exemplars classifier (RISE) was introduced by Pedro Domingos from University of California in 1996 [1]. The most innovative feature of this algorithm is that it combines rule induction with instance-based learning, in the attempt of building a more accurate classifier than previous rule-based approaches.

The objective of this work is the implementation and evaluation of a custom Python version of RISE for classification tasks, applying it to 4 data sets of different sizes and features. The accuracy of each trained model is computed, as well as the coverage and local accuracy of each rule.

[1] Domingos, P. (1996). Unifying instance-based and rule-based induction. Machine Learning, 24(2), 141-168.

| | |
|-|-|
| **Author** | Albert Espín (except datasets, gathered from [UCI's Machine Learning repository](http://archive.ics.uci.edu/ml)) |
| **Date**  | March 2019  |
| **Code license**  | MIT |
| **Report license**  | Creative Commons Attribution, Non-Commercial, Non-Derivative |
|**Dataset licence** | Licenses specified for each dataset in [UCI's Machine Learning repository](http://archive.ics.uci.edu/ml) |


## Pseudo-code

![](pseudo_code.png)


