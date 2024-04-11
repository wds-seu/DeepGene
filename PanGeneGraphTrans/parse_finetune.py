from modeling_roformer import *


def parse_finetune_method(args):
    param_file = args.model_dir + 'pretrain_params_epoch_' + str(args.load_epoch)
    model = RoFormerForSequenceClassification.from_pretrained(param_file, num_labels=args.class_number)
    return model


def parser_finetune_add_main_args(parser):
    parser.add_argument('--class_number', type=int, default=2)
    parser.add_argument('--metric', type=str, default='mcc', choices=['mcc', 'f1'],
                        help='evaluation metric')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--data_dir', type=str, default='../data/LPD/promoter_prediction/prom_300/')
    parser.add_argument('--model_dir', type=str, default='../model/')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval_step', type=int,
                        default=200, help='how often to print')
    parser.add_argument('--load_model', action='store_true', help='whether to load model')
    parser.add_argument('--load_epoch', type=int, default=20)

    # hyper_parameter for model arch and training
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
