import argparse
from tqdm import tqdm
import torch
from os import listdir
from os.path import isfile, join


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help='Model to use for explanations.')
    parser.add_argument('--dataset', default="", help='Dataset to explain.')
    parser.add_argument('--expl', default="", help='Explainer to use.')
    args = parser.parse_args()

    torch.manual_seed(42)

    pretrained_models = [f.split(".")[0] for f in listdir("Models/pretrained") if isfile(join("Models/pretrained", f))]
    
    assert args.model + "_" + args.dataset in pretrained_models
    
