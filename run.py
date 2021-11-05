import argparse
from types import TracebackType
import yaml
from solver import Solver

def get_parser():
    parser = argparse.ArgumentParser(description="sEMG-based gesture recognition")
    parser.add_argument(
        "--config",
        "-cfg",
        default="./config/inter_session/capgmyo_dbb.yaml",
        type=str,
        help="Config file which is used.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["intra_session", "inter_session", "inter_subject"],
        default="inter_session",
        help="Choose your task: 1. intra_session  2. inter_session  3. inter_subject")
    parser.add_argument(
        "--stage",
        "-sg",
        type=str,
        choices=["pretrain", "train", "test"],
        default="pretrain",
        help="Choose your stage: 1. pretrain  2. train  3. test")
    
    # contrastive
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')

    # yaml args
    parser.add_argument("--subjects", "-s", nargs="*", type=int, default=None)
    parser.add_argument("--num_epochs", "-ne", type=int, default=None)
    parser.add_argument("--batch_size", "-bs", type=int, default=None)
    parser.add_argument("--window_size", "-wz", type=int, default=None)
    parser.add_argument("--window_step", "-ws", type=int, default=None)

    return parser


def update_args():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        # default_arg = yaml.load(f, Loader=yaml.FullLoader)
        default_arg = yaml.load(f)
    # print(args)
    # print(vars(args))
    # print(default_arg)
    

    # update args if specified on the command line
    args = vars(args)

    stage = args["stage"] # pretrain, train, test
    stage_args = {}
    extra_args = {}

    keys = list(args.keys())
    for key in keys:
        if key in ["subjects", "num_epochs", "batch_size", "window_size", "window_step"]:
            if args[key]:
                stage_args[key] = args[key]
        else:
            extra_args[key] = args[key]

    # print(stage_args)
    # print(extra_args)

    default_arg[stage].update(stage_args)
    default_arg.update(extra_args)
    # for k, v in default_arg.items():
    #     print(k, ": ", v)

    parser.set_defaults(**default_arg)
    args = parser.parse_args()
    # for k, v in vars(args).items():
        # print(k, ": ", v)
    # print(yaml.dump(vars(args)))

    return args

if __name__ == "__main__":
    args = update_args()
    solver = Solver(args)
    solver.start()
