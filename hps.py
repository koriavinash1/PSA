import argparse

HPARAMS_REGISTRY = {}


class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)




clevr = Hparams()
clevr.learning_rate = 2e-4
clevr.batch_size = 32
clevr.weight_decay = 0.01
clevr.input_res = 64
clevr.pad = 4
clevr.nslots = 7
clevr.nconditions = 0
clevr.max_num_obj = 7
clevr.enc_arch = "64b1d2,32b1d2,16b1d2,8b1"
clevr.dec_arch = "8b1u2,16b1u2,32b1u2,64b1"
clevr.channels = [16, 32, 64, 128]
HPARAMS_REGISTRY["clevr"] = clevr


clevr_hans = Hparams()
clevr_hans.update(clevr.__dict__)
clevr_hans.nconditions = 19
clevr_hans.max_num_obj = 10
clevr_hans.nslots = 10
clevr_hans.data_dir = '/vol/biomedic2/agk21/PhDLogs/datasets/CLEVR/CLEVR-Hans3'
HPARAMS_REGISTRY["clevr_hans"] = clevr_hans



def setup_hparams(parser: argparse.ArgumentParser) -> Hparams:
    hparams = Hparams()
    args = parser.parse_known_args()[0]
    valid_args = set(args.__dict__.keys())
    

    if (args.nconditions > 0) and (args.model == 'SSAU'):
        args.hps = "clevr_hans"

    hparams_dict = HPARAMS_REGISTRY[args.hps].__dict__
    for k in hparams_dict.keys():
        if k not in valid_args:
            raise ValueError(f"{k} not in default args")
    parser.set_defaults(**hparams_dict)
    hparams.update(parser.parse_known_args()[0].__dict__)
    return hparams


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--exp_name", 
        help="Experiment name.", 
        type=str, default=""
    )
    parser.add_argument(
        "--data_dir", 
        help="Data directory to load form.", 
        type=str, default="/data2"
    )
    parser.add_argument(
        "--max_num_obj", 
        type=int, default=6
    )

    parser.add_argument(
        "--hps", 
        help="hyperparam set.", 
        type=str, default="clevr"
    )

    parser.add_argument(
        "--resume", 
        help="Path to load checkpoint.", 
        type=str, default=""
    )
    parser.add_argument(
        "--seed", 
        help="Set random seed.", 
        type=int, default=7
    )
    parser.add_argument(
        "--deterministic",
        help="Toggle cudNN determinism.",
        action="store_true",
        default=True,
    )
    # training
    parser.add_argument(
        "--epochs", 
        help="Training epochs.", 
        type=int, default=500
    )
    parser.add_argument(
        "-bs",
        "--batch_size", 
        help="Batch size.", 
        type=int, default=32
    )
    parser.add_argument(
        "-lr", 
        "--learning_rate",
        help="Learning rate.", 
        type=float, default=1e-3
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        help="lr warmup steps.", 
        type=int, default=100000
    )
    parser.add_argument(
        "-wd",
        "--weight_decay", 
        help="Weight decay penalty.", 
        type=float, default=0.01
    )
    parser.add_argument(
        "--betas",
        help="Adam beta parameters.",
        nargs="+",
        type=float,
        default=[0.9, 0.9],
    )
    parser.add_argument(
        "--ema_rate", 
        help="Exp. moving avg. model rate.", 
        type=float, default=0.999
    )
    parser.add_argument(
        "--input_res", 
        help="Input image crop resolution.", 
        type=int, default=64
    )
    parser.add_argument(
        "--input_channels", 
        help="Input image num channels.", 
        type=int, default=3
    )
    parser.add_argument(
        "--pad", 
        help="Input padding.", 
        type=int, default=3
    )
    parser.add_argument(
        "--hflip", 
        help="Horizontal flip prob.", 
        type=float, default=0.5
    )
    parser.add_argument(
        "--grad_clip", 
        help="Gradient clipping value.", 
        type=float, default=200
    )
    parser.add_argument(
        "--grad_skip", 
        help="Skip update grad norm threshold.", 
        type=float, default=55000
    )
    parser.add_argument(
        "--accu_steps", 
        help="Gradient accumulation steps.", 
        type=int, default=1
    )
    parser.add_argument(
        "--beta", 
        help="Max KL beta penalty weight.", 
        type=float, default=1.0
    )
    parser.add_argument(
        "--beta_warmup_steps", 
        help="KL beta penalty warmup steps.", 
        type=int, default=0
    )
    parser.add_argument(
        "--kl_free_bits", 
        help="KL min free bits constraint.", 
        type=float, default=0.0
    )
    parser.add_argument(
        "--viz_freq", 
        help="Steps per visualisation.", 
        type=int, default=10000
    )
    parser.add_argument(
        "--eval_freq", 
        help="Train epochs per validation.", 
        type=int, default=5
    )

    # model
    parser.add_argument(
        "--model",
        help="SA model: SA/ASA/VSA/VASA/SSA/SSAU.",
        type=str,
        default="SA",
    )
    parser.add_argument(
        "--enc_arch",
        help="Encoder architecture config.",
        type=str,
        default="64b1d2,32b1d2,16b1d2,8b1d8",
    )
    parser.add_argument(
        "--dec_arch",
        help="Decoder architecture config.",
        type=str,
        default="8b1,16b1u2,32b1u2,64b1u2",
    )
    parser.add_argument(
        "--channels",
        help="Number of channels.",
        nargs="+",
        type=int,
        default=[16, 32, 48, 64, 128],
    )
    parser.add_argument(
        "--bottleneck", 
        help="Bottleneck width factor.", 
        type=int, default=4
    )
    
    parser.add_argument(
        "--bias_max_res",
        help="Learned bias param max resolution.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--x_like",
        help="x likelihood: {fixed/shared/diag}_{gauss/dgauss}.",
        type=str,
        default="diag_dgauss",
    )
    parser.add_argument(
        "--std_init",
        help="Initial std for x scale. 0 is random.",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--ckpt_dir",
        help="Directory to save logs.",
        type=str,
        default='./checkpoints',
    )


    # latent args
    parser.add_argument(
        "--zprior", 
        help="latent prior selection: {gmm/gauss}.",
        type=str,
        default='gauss')
    parser.add_argument(
        "--num_components", 
        help='If zprior is gmm, number of components',
        type=int,
        default=50)
    parser.add_argument(
        "--learn_prior", 
        help="learn/fix the prior distribution.",
        action="store_true",
        default=False)


    # SA args
    parser.add_argument(
        "--niters", 
        type=int,
        default=3)
    parser.add_argument(
        "--nslots", 
        type=int,
        default=7)
    parser.add_argument(
        "--slot_dim", 
        type=int,
        default=128)
    parser.add_argument(
        "--implicit", 
        action="store_true",
        default=False)
    parser.add_argument(
        "--nconditions", 
        type=int,
        default=19)
    parser.add_argument(
        "--initial_decoder_spatial_resolution", 
        type=int,
        default=8)
    return parser