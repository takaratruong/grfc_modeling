import wandb
from algs.FC_Net import FC_Net

if __name__ == '__main__':

    model = FC_Net()

    print(model)
    model.train()

