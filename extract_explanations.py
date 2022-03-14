import argparse
import torch
from os import listdir
from os.path import isfile, join

import Models.models_CORA as models_CORA
import Models.models_MUTAG as models_MUTAG
import Models.models_REDDIT as models_REDDIT
import Models.models_BAshapes as models_BAshapes
import Models.models_CITESEER as models_CITESEER

from Explainers.explainers import * #SemiSupSubGraphX, SemiSupPGExplainer, LargeSemiSupPGExplainer


def explain_SubGraphX(model, dataset, dataset_name, model_name):
    fw = SemiSupSubGraphX(model, dataset, dataset_name, model_name)
    fw.explain(max_nodes=10)


def explain_PGExplainer(framework, dataset_name, model_name, save, num_epochs):
    if dataset_name == "REDDIT":
        fw = LargeSemiSupPGExplainer(framework, dataset_name, model_name, num_epochs=num_epochs, num_hops=3)
    else:
        fw = SemiSupPGExplainer(framework, dataset_name, model_name, num_epochs=num_epochs, num_hops=3)
    fw.explain(top_k=5, save=save)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help='Model to use for explanations.')
    parser.add_argument('--dataset', default="", help='Dataset to explain.')
    parser.add_argument('--expl', default="", help='Explainer to use.')
    parser.add_argument('--save', action='store_true', default=False, help='Whether to save the trained model or not.')
    args = parser.parse_args()
    print(f"You are{' not' if not args.save else ''} saving the result")

    torch.manual_seed(42)

    pretrained_models = [f.split(".")[0] for f in listdir("Pretrained models") if isfile(join("Pretrained models", f))]
    explainers = ["subgraphx", "pgexplainer"]
    
    assert args.model + "_" + args.dataset in pretrained_models , "Model not yet implemented or trained"
    assert args.expl.lower() in explainers , "Explainer not yet implemented"

    num_epochs = 20
    if args.dataset.upper() == "CORA":
        num_epochs = 20
        fw = models_CORA.getFrameworkByName(args.model.upper())
    elif args.dataset.upper() == "MUTAG":
        fw = models_MUTAG.getFrameworkByName(args.model.upper())
    elif args.dataset.upper() == "REDDIT":
        num_epochs = 3
        fw = models_REDDIT.getFrameworkByName(args.model.upper(), batch_size=512, num_workers=4)
    elif args.dataset.upper() == "BASHAPES":
        num_epochs = 30 #check file for correct number of epochs
        fw = models_BAshapes.getFrameworkByName(args.model.upper())
    elif args.dataset.upper() == "CITESEER":
        num_epochs = 20
        fw = models_CITESEER.getFrameworkByName(args.model.upper())
    fw.load_model()

    if args.expl.lower() == "subgraphx":
        explain_SubGraphX(fw.model, fw.dataset, args.dataset.upper(), args.model.upper())
    elif args.expl.lower() == "pgexplainer":
        explain_PGExplainer(fw, args.dataset.upper(), args.model.upper(), args.save, num_epochs)
