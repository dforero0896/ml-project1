# Notes
Write your notes, tests done, preliminary results, etc.

## oct 9

+ No cleaning test data (we would lose data lines which are checked by submission platform).
+ No cleaning train data.
+ Standardizing test data matrix (tx).
+ No standardizing test labels (we are provided a function to cast our predictions into +/-1)
+ Submitted predictions with all methids implemented so far (GD adapting step, SGD adapting step, Lsq, RR), got ~65% accuracy.
+ No adapting the step yields comparable results.
+ Saving all weight from SGD and choosing the ones with the least loss?
+ TODO: Add offset to feature matrix (column of 1's)
