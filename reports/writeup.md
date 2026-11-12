# Predicting Gas Turbine Wear from Sensor Data: Write-up

## 1. Summary
problem, approach, best model, important numbers and such

## 2. Dataset and cleaning
source, size, columns and cleaning. also include steps for cleaning, missing-value check, 2 constant columns dropped

## 3. Exploration and key findings
a couple of findings with a figure to support (figures 01 to 04). maybe throttle being extremely dominant, wear visible at a fixed throttle, different sensors track different wear?

## 4. Preprocessing
80/20 split, seed 42, no scaling or encoding needed, lever position kept as an input

## 5. Modeling strategy
explain tested models and why? (baseline, linear regression, decision tree, random forest) default settings, no tuning done

## 6. Model comparison
results table, figure 05, which model won and why, potential pros and cons/tradeoffs between them? train vs test error for the tree and forest

## 7. Error analysis
figure 06 to find where predictions stray from the actual values by target and by throttle setting.

## 8. Feature importance
figure 07 to find the top sensors, whether they match section 3, downsides (combined targets, possible near-duplicate sensors)

## 9. Takeaways
explain results in more understandable way

## 10. Limitations and future work
limitations of the project (simulated data, random split on a grid, one seed, default settings) and 2 or 3 next steps.*