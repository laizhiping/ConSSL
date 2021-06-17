import argparse
from types import TracebackType
import yaml
from solver import Solver

def get_parser():
    parser = argparse.ArgumentParser(description="sEMG-based gesture recognition")
    parser.add_argument(
        "--config",
        "-cfg",
        default="./config/inter-session/capgmyo-dbb.yaml",
        type=str,
        help="Config file which is used.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["intra-session", "inter-session", "inter-subject"],
        default="inter-session",
        help="Choose your task: 1. intra-session  2. inter-session  3. inter-subject")
    parser.add_argument(
        "--stage",
        "-sg",
        type=str,
        choices=["pretrain", "train", "test"],
        default="pretrain",
        help="Choose your stage: 1. pretrain  2. train  3. test")

    # yaml args
    parser.add_argument("--subjects", "-s", nargs="*", type=int, default=None)
    parser.add_argument("--num_epochs", "-ne", type=int, default=None)
    parser.add_argument("--batch_size", "-bs", type=int, default=None)
    parser.add_argument("--window_size", "-wz", type=int, default=None)
    parser.add_argument("--window_step", "-ws", type=int, default=None)
    
    # contrastive
    parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        # default_arg = yaml.load(f, Loader=yaml.FullLoader)
        default_arg = yaml.load(f)

    # update args if specified on the command line
    args = vars(args)
    keys = list(args.keys())
    for key in keys:
        if args[key] is None:
            del args[key]
    default_arg.update(args)
    parser.set_defaults(**default_arg)

    args = parser.parse_args()
    # for k, v in vars(args).items():
    #     print(k, ": ", v)
    # print(yaml.dump(vars(args)))
    solver = Solver(args)
    solver.start(task=args.task)
