import argparse

parser = argparse.ArgumentParser()

# general args
parser.add_argument("--use_cuda", type=int,
                    help="try to use cuda if positive",
                    default=0)
parser.add_argument('--device', type=int,
                    help="GPU device number",
                    default=0)
parser.add_argument("--lr", type=float,
                    help="learning rate",
                    default=2e-4)
parser.add_argument("--n_epoch", type=int,
                    help="max number of epochs",
                    default=20)
parser.add_argument("--gpath", type=str,
                    help="local graph path to load",
                    default='typedEntGrDirC_3_3')
parser.add_argument("--gload_path", type=str,
                    help="global graph path to load and start training from",
                    default='NONE')
parser.add_argument("--writepath", type=str,
                    help="global graph path to save",
                    default='typedEntGrDir_3_3_gf')
parser.add_argument("--lambda1", type=float,
                    help="normalization factor of |W|_1",
                    default=0.01)
parser.add_argument("--featIdx", type=int,
                    help="index of score feature in local graphfile",
                    default=4)
parser.add_argument("--threshold", type=float,
                    help="lowest edge weight to retain",
                    default=0.01)
parser.add_argument("--CCG", type=int,
                    help="Whether it's CCG parsing or openIE")
parser.add_argument("--maxRank", type=int,
                    help="maximum number of neighbors to read for a node")
parser.add_argument("--lambda_trans", type=float,
                    help="factor of transitivity loss",
                    default=0.)
parser.add_argument("--epsilon_trans", type=float,
                    help="epsilon of transitivity loss",
                    default=0.98)
parser.add_argument("--trans_method", type=int,
                    help="the function index for transtivity loss, 1:log wij+log wjk -> log wik, 2:I(wijwjk>wik)log wik, 3:I(wijwjk>wik)wijwjk*log wik",
                    default=0)

args = parser.parse_args()