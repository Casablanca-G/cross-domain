def get_main_arguments(parser):
    """Required parameters"""

    parser.add_argument("--model_name",
                        default='llm4cdsr',
                        choices=['sasrec_seq', 'llm4cdsr', 'ibml_cdsr'],
                        type=str,
                        required=False,
                        help="model name")
    parser.add_argument("--dataset", 
                        default="douban", 
                        choices=["douban", "amazon", "elec", # preprocess by myself
                                ], 
                        help="Choose the dataset")
    parser.add_argument("--domain",
                        default="0",
                        type=str,
                        help="the domain flag for SDSR")
    parser.add_argument("--inter_file",
                        default="book_movie",
                        type=str,
                        help="the name of interaction file")
    parser.add_argument("--pretrain_dir",
                        type=str,
                        default="sasrec_seq",
                        help="the path that pretrained model saved in")
    parser.add_argument("--output_dir",
                        default='./saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--check_path",
                        default='',
                        type=str,
                        help="the save path of checkpoints for different running")
    parser.add_argument("--do_test",
                        default=False,
                        action="store_true",
                        help="whehther run the test on the well-trained model")
    parser.add_argument("--do_emb",
                        default=False,
                        action="store_true",
                        help="save the user embedding derived from the SRS model")
    parser.add_argument("--do_group",
                        default=False,
                        action="store_true",
                        help="conduct the group test")
    parser.add_argument("--do_cold",
                        default=False,
                        action="store_true",
                        help="whether test cold start")
    parser.add_argument("--ts_user",
                        type=int,
                        default=10,
                        help="the threshold to split the short and long seq")
    parser.add_argument("--ts_item",
                        type=int,
                        default=20,
                        help="the threshold to split the long-tail and popular items")
    
    return parser


def get_model_arguments(parser):
    """Model parameters"""
    
    parser.add_argument("--hidden_size",
                        default=64,
                        type=int,
                        help="the hidden size of embedding")
    parser.add_argument("--trm_num",
                        default=2,
                        type=int,
                        help="the number of transformer layer")
    parser.add_argument("--num_heads",
                        default=1,
                        type=int,
                        help="the number of heads in Trm layer")
    parser.add_argument("--num_layers",
                        default=1,
                        type=int,
                        help="the number of GRU layers")
    parser.add_argument("--cl_scale",
                        type=float,
                        default=0.1,
                        help="the scale for contastive loss")
    parser.add_argument("--tau",
                        default=1,
                        type=float,
                        help="the temperature for contrastive loss")
    parser.add_argument("--tau_reg",
                        default=1,
                        type=float,
                        help="the temperature for contrastive loss")
    parser.add_argument("--dropout_rate",
                        default=0.5,
                        type=float,
                        help="the dropout rate")
    parser.add_argument("--max_len",
                        default=200,
                        type=int,
                        help="the max length of input sequence")
    parser.add_argument("--mask_prob",
                        type=float,
                        default=0.6,
                        help="the mask probability for training Bert model")
    parser.add_argument("--mask_crop_ratio",
                        type=float,
                        default=0.3,
                        help="the mask/crop ratio for CL4SRec")
    parser.add_argument("--aug",
                        default=False,
                        action="store_true",
                        help="whether augment the sequence data")
    parser.add_argument("--aug_seq",
                        default=False,
                        action="store_true",
                        help="whether use the augmented data")
    parser.add_argument("--aug_seq_len",
                        default=0,
                        type=int,
                        help="the augmented length for each sequence")
    parser.add_argument("--aug_file",
                        default="inter",
                        type=str,
                        help="the augmentation file name")
    parser.add_argument("--train_neg",
                        default=1,
                        type=int,
                        help="the number of negative samples for training")
    parser.add_argument("--test_neg",
                        default=100,
                        type=int,
                        help="the number of negative samples for test")
    parser.add_argument("--suffix_num",
                        default=5,
                        type=int,
                        help="the suffix number for augmented sequence")
    parser.add_argument("--prompt_num",
                        default=2,
                        type=int,
                        help="the number of prompts")
    parser.add_argument("--freeze",
                        default=False,
                        action="store_true",
                        help="whether freeze the pretrained architecture when finetuning")
    parser.add_argument("--freeze_emb",
                        default=False,
                        action="store_true",
                        help="whether freeze the embedding layer, mainly for LLM embedding")
    parser.add_argument("--alpha",
                        default=0.1,
                        type=float,
                        help="the weight of auxiliary loss")
    parser.add_argument("--beta",
                        default=0.1,
                        type=float,
                        help="the weight of regulation loss")
    parser.add_argument("--llm_emb_file",
                        default="item_emb",
                        type=str,
                        help="the file name of the LLM embedding")
    parser.add_argument("--expert_num",
                        default=1,
                        type=int,
                        help="the number of adapter expert")
    parser.add_argument("--user_emb_file",
                        default="usr_profile_emb",
                        type=str,
                        help="the file name of the user LLM embedding")
    # for LightGCN
    parser.add_argument("--layer_num",
                        default=2,
                        type=int,
                        help="the number of collaborative filtering layers")
    parser.add_argument("--keep_rate",
                        default=0.8,
                        type=float,
                        help="the rate for dropout")
    parser.add_argument("--reg_weight",
                        default=1e-6,
                        type=float,
                        help="the scale for regulation of parameters")
    # for LLM4CDSR
    parser.add_argument("--local_emb",
                        default=False,
                        action="store_true",
                        help="whether use the LLM embedding to initilize the local embedding")
    parser.add_argument("--global_emb",
                        default=False,
                        action="store_true",
                        help="whether use the LLM embedding to substitute global embedding")
    parser.add_argument("--thresholdA",
                        default=0.5,
                        type=float,
                        help="mask rate for AMID")
    parser.add_argument("--thresholdB",
                        default=0.5,
                        type=float,
                        help="mask rate for AMID")
    parser.add_argument("--hidden_size_attr",
                        default=32,
                        type=int,
                        help="the hidden size of attribute embedding")

    # ===== Selective Learning (PR-2) =====
    parser.add_argument("--use_sl",
                        default=False,
                        action="store_true",
                        help="enable Selective Learning dual-mask mechanism")
    parser.add_argument("--sl_use_unc",
                        default=True,
                        type=lambda x: x.lower() not in ("false", "0", "no"),
                        help="enable the Uncertainty Mask branch of SL (default True; pass False to disable)")
    parser.add_argument("--sl_use_ano",
                        default=False,
                        type=lambda x: x.lower() not in ("false", "0", "no"),
                        help="enable the Anomaly Mask branch of SL (default False; pass True to enable)")
    parser.add_argument("--sl_ru",
                        default=0.10,
                        type=float,
                        help="drop top r_u%% positions ranked by predictive entropy")
    parser.add_argument("--sl_ra",
                        default=0.10,
                        type=float,
                        help="drop bottom r_a%% positions ranked by anomaly score (PR-3)")
    parser.add_argument("--sl_warmup_epochs",
                        default=5,
                        type=int,
                        help="warmup epochs during which SL is disabled")
    parser.add_argument("--sl_combine",
                        default="and",
                        choices=["and", "or"],
                        type=str,
                        help="how to combine uncertainty/anomaly masks")
    parser.add_argument("--sl_entropy_on",
                        default="candidates",
                        choices=["candidates", "full"],
                        type=str,
                        help="compute entropy over pos+neg candidates or the full item vocab")
    parser.add_argument("--sl_g_lambda",
                        default=0.1,
                        type=float,
                        help="weight of the trainable g(.) anomaly-estimator auxiliary loss")

    # ===== IBML (Information-Balanced Multimodal Learning) =====
    parser.add_argument("--ibml_use_bio", default=True,
                        type=lambda x: x.lower() not in ("false", "0", "no"),
                        help="enable Balance Information Optimization (BIO) loss reweighting")
    parser.add_argument("--ibml_use_tcm", default=True,
                        type=lambda x: x.lower() not in ("false", "0", "no"),
                        help="enable Task Complexity Modulation (input noise on dominant domain)")
    parser.add_argument("--ibml_momentum", default=0.9, type=float,
                        help="EMA momentum for rho^d tracking")
    parser.add_argument("--ibml_warmup", default=3, type=int,
                        help="epochs before IBML kicks in (pure baseline loss during warmup)")
    parser.add_argument("--ibml_lambda_fusion", default=1.0, type=float,
                        help="scale of the fusion-pathway (AB) loss term in the BIO objective")
    parser.add_argument("--ibml_lambda_bio", default=1.0, type=float,
                        help="scale of the per-domain upweighting term when a domain is lazy")
    parser.add_argument("--ibml_tcm_alpha", default=1.0, type=float,
                        help="TCM noise sensitivity: mask_ratio = alpha*(rho^d - rho_bar)")
    parser.add_argument("--ibml_tcm_max", default=0.4, type=float,
                        help="upper bound on the TCM mask ratio")
    parser.add_argument("--ibml_gap_eps", default=1e-3, type=float,
                        help="minimum rho gap before a domain is flagged as under-optimised")
    return parser


def get_train_arguments(parser):
    """Training parameters"""
    
    parser.add_argument("--train_batch_size",
                        default=512,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--lr",
                        default=0.001,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--l2",
                        default=0,
                        type=float,
                        help='The L2 regularization')
    parser.add_argument("--num_train_epochs",
                        default=100,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--lr_dc_step",
                        default=1000,
                        type=int,
                        help='every n step, decrease the lr')
    parser.add_argument("--lr_dc",
                        default=0,
                        type=float,
                        help='how many learning rate to decrease')
    parser.add_argument("--patience",
                        type=int,
                        default=20,
                        help='How many steps to tolerate the performance decrease while training')
    parser.add_argument("--watch_metric",
                        type=str,
                        default='NDCG@10',
                        help="which metric is used to select model.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for different data split")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gpu_id',
                        default=0,
                        type=int,
                        help='The device id.')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='The number of workers in dataloader')
    parser.add_argument("--log", 
                        default=False,
                        action="store_true",
                        help="whether create a new log file")
    
    return parser
