# Task 1
Plotting can be a bit of a bottomless well, but I found KDE plots of comparing patients with disease to the control useful.
As far as I understand, the Filtered entries could both indicate low intensities, but also high uncertainty or similar.
I think transformers should be good as a classifier since they support masked features, but I opted for a simple random forest model, as it allows for direct estimation of feature importance.
Here the Filtered entries was just replaced with a small value, since the random forest architecture should be robust under this assumption (or atleast more robust than non-tree methods).

## Feature importance
It is likely better to do some initial pruning and use a permutation-based importance metric, especially since the classifier does a decent job at ~84%.
But the simpler Gini-importance is much faster, so will have to do for this purpose.
The 10 most important features are plotted in the [plots](./plots) folder.
