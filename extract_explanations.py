import argparse
import torch
from os import listdir
from os.path import isfile, join

import Models.models_CORA as models_CORA
import Models.models_MUTAG as models_MUTAG
import Models.models_REDDIT as models_REDDIT

from Explainers.explainers import SemiSupSubGraphX


def explain_SubGraphX(model, dataset, dataset_name, model_name):
    fw = SemiSupSubGraphX(model, dataset, dataset_name, model_name)
    fw.explain(max_nodes=10)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help='Model to use for explanations.')
    parser.add_argument('--dataset', default="", help='Dataset to explain.')
    parser.add_argument('--expl', default="", help='Explainer to use.')
    args = parser.parse_args()

    torch.manual_seed(42)

    pretrained_models = [f.split(".")[0] for f in listdir("Models/pretrained") if isfile(join("Models/pretrained", f))]
    explainers = ["subgraphx"]
    
    assert args.model + "_" + args.dataset in pretrained_models , "Model not yet implemented or trained"
    assert args.expl.lower() in explainers , "Explainer not yet implemented"


    if args.dataset.upper() == "CORA":
        fw = models_CORA.getFrameworkByName(args.model.upper())
    elif args.dataset.upper() == "MUTAG":
        fw = models_MUTAG.getFrameworkByName(args.model.upper())
    elif args.dataset.upper() == "REDDIT":
        fw = models_REDDIT.getFrameworkByName(args.model.upper())

    if args.expl == "subgraphx":
        explain_SubGraphX(fw.model, fw.dataset, args.dataset.upper(), args.model.upper())
    
