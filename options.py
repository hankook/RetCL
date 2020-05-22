
def add_model_arguments(parser):
    parser.add_argument('--module', type=str, default='v1')
    parser.add_argument('--encoder', type=str, default='s2v')
    parser.add_argument('--num-layers', type=int, default=5)
    parser.add_argument('--num-branches', type=int, default=2)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--bn-first', action='store_true')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--use-label', action='store_true')
