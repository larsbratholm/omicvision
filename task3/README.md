# Task 3
The optimization of drug doses is treated as a constrained minimization:

```
min sum((diseased - healthy + EffectMatrix @ doses)**2)
```
under the constraint that the total toxicity is 10 or less.

The squared loss without treatment is 1802.
Restricting the solutions to integer doses yields a loss of 1794 with a single dose of Drug4, while lifting this restriction yielsd a loss of 1695.8 with weights:
[7.17821955e-02 1.77623479e-01 1.55445614e-07 6.28236810e-01
 1.78289835e-01 7.00084770e-02 2.69112367e-08 5.15402237e-01
 2.33520796e-02 1.41513541e-01]
