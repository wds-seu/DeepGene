from modeling_roformer import *


def parse_method(args):
    if args.load_model:
        param_file = args.model_dir + 'pretrain_params_epoch_' + str(args.load_epoch)
        model = RoFormerForMaskedLM.from_pretrained(param_file)
    else:
        config = RoFormerConfig()
        model = RoFormerForMaskedLM(config)
    return model


def parser_add_main_args(parser):
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--data_dir', type=str, default='../data/pretrain/')
    parser.add_argument('--model_dir', type=str, default='../model/')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--save_step', type=int,
                        default=5, help='how often to save model')
    parser.add_argument('--load_model', action='store_true', help='whether to load model')
    parser.add_argument('--load_epoch', type=int, default=0)

    # hyper_parameter for model arch and training
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--subgraph_size', type=int, default=128)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)
