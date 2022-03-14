# Advanced Topics in Machine Learning and Optimization course project

This project is the final course project for the aforementioned course @ UNITN.

A detailed description of the project can be found in `Report.pdf`

## Requisites
- Create a *tmp* folder in the root directory
- Import the conda env *env.yml*
- Install this modified version of [DIG](https://github.com/steveazzolin/dive-into-graphs)

## Train a GNN
For example, to train a GCN over the CORA dataset:
```
cd Models
python models_CORA.py --model=GCN --train --save
```

By specifying also `--wandb` the logging to wanb is enabled

To run a trained model instead:
```
cd Models
python models_CORA.py --model=GCN
```

## Extract explanation
```
python extract_explanations.py --model=GCN --dataset=CORA --expl=PGExplainer --save
```

## Pattern mining
To extract the features of the local explanations, cutting irrelevant edges and removing the connected components not including the target node to be explained:
```
python mine_explanations.py --model=GCN --dataset=CORA --expl=PGExplainer --cut_edges --cut_cc
```
The code above can also be sued to plot the local explanations, to plot the prototypes as found by the greedy Edit distance based algorithm, and to plot the edge scores distribution.
