import argparse
import traceback
import logging
import yaml
import sys
import os
import numpy as np
import time
from glob import glob
from tqdm import tqdm
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu

from diffusion.script_util import IDDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import sa_loss
from losses.clip_module import CLIP_Module
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment

class Generation(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]

    def finetune(self):
        print(self.args.exp)
        print(f'   {self.src_txts}')
        print(f'-> {self.trg_txts}')

        # ----------- Model -----------#
        if self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ"]:
            model = IDDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optim_ft, step_size=1, gamma=self.args.sch_gamma)
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Loss -----------#
        print("Loading losses")
        clip_func = CLIP_Module(
            self.device,
            lambda_dcl=1,
            lambda_cgl=0,
            clip_model=self.args.clip_model_name)
        sa_loss_func = sa_loss.SALoss().to(self.device).eval()

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}
        for mode in ['train', 'test']:
            img_lat_pairs = []
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                print('DATASET_PATHSssssssss', DATASET_PATHS, self.config.data.dataset)
                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, img in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma)
                            progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            model.module.load_state_dict(init_ckpt)
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)
            clip_func.target_direction = None

            # ----------- Train -----------#
            for it_out in range(self.args.n_iter):
                exp_id = os.path.split(self.args.exp)[-1]
                save_name = f'checkpoint/{exp_id}_{trg_txt.replace(" ", "_")}-{it_out}.pth'
                if self.args.do_train:
                    if os.path.exists(save_name):
                        print(f'{save_name} already exists.')
                        model.module.load_state_dict(torch.load(save_name))
                        continue
                    else:
                        for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic['train']):
                            model.train()
                            time_in_start = time.time()

                            optim_ft.zero_grad()
                            x = x_lat.clone()

                            with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                                for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    progress_bar.update(1)

                            cgl_dcl_loss = (2 - clip_func(x0, src_txt, xz, trg_txt)) / 2
                            cgl_dcl_loss = -torch.log(cgl_dcl_loss)
                            loss_sal = torch.mean(sa_loss_func(x0, x))
                            loss_l1 = nn.L1Loss()(x0, x)
                            loss = self.args.cgl_dcl_loss_w * cgl_dcl_loss + self.args.sa_loss_w * loss_sal + self.args.l1_loss_w * loss_l1
                            loss.backward()

                            optim_ft.step()
                            print(f"CLIP {step}-{it_out}: loss_sal: {loss_sal:.3f}, cgl_dcl_loss: {cgl_dcl_loss:.3f}")

                            if self.args.save_train_image:
                                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                           f'train_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png'))
                            time_in_end = time.time()
                            print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")
                            if step == self.args.n_train_img - 1:
                                break

                        if isinstance(model, nn.DataParallel):
                            torch.save(model.module.state_dict(), save_name)
                        else:
                            torch.save(model.state_dict(), save_name)
                        print(f'Model {save_name} is saved.')
                        scheduler_ft.step()

                # ----------- Eval -----------#
                if self.args.do_test:
                    if not self.args.do_train:
                        print(save_name)
                        model.module.load_state_dict(torch.load(save_name))

                    model.eval()
                    img_lat_pairs = img_lat_pairs_dic[mode]
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):
                        with torch.no_grad():
                            x = x_lat
                            with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    progress_bar.update(1)

                            print(f"Eval {step}-{it_out}")
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'{mode}_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_test_step}.png'))
                            if step == self.args.n_test_img - 1:
                                break

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # Mode
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='./runs/', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--ni', type=int, default=1,  help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--align_face', type=int, default=1, help='align face or not')

    # Text
    parser.add_argument('--edit_attr', type=str, default=None, help='Attribute to edit defiend in ./utils/text_dic.py')
    parser.add_argument('--src_txts', type=str, action='append', help='Source text e.g. Face')
    parser.add_argument('--trg_txts', type=str, action='append', help='Target text e.g. Angry Face')
    parser.add_argument('--target_class_num', type=str, default=None)

    # Sampling
    parser.add_argument('--t_0', type=int, default=400, help='Return step in [0, 1000)')
    parser.add_argument('--n_inv_step', type=int, default=40, help='# of steps during generative pross for inversion')
    parser.add_argument('--n_train_step', type=int, default=6, help='# of steps during generative pross for train')
    parser.add_argument('--n_test_step', type=int, default=40, help='# of steps during generative pross for test')
    parser.add_argument('--sample_type', type=str, default='ddim', help='ddpm for Markovian sampling, ddim for non-Markovian sampling')
    parser.add_argument('--eta', type=float, default=0.0, help='Controls of varaince of the generative process')

    # Train & Test
    parser.add_argument('--do_train', type=int, default=1, help='Whether to train or not during CLIP finetuning')
    parser.add_argument('--do_test', type=int, default=1, help='Whether to test or not during CLIP finetuning')
    parser.add_argument('--save_train_image', type=int, default=1, help='Wheter to save training results during CLIP fineuning')
    parser.add_argument('--bs_train', type=int, default=1, help='Training batch size during CLIP fineuning')
    parser.add_argument('--bs_test', type=int, default=1, help='Test batch size during CLIP fineuning')
    parser.add_argument('--n_precomp_img', type=int, default=100, help='# of images to precompute latents')
    parser.add_argument('--n_train_img', type=int, default=50, help='# of training images')
    parser.add_argument('--n_test_img', type=int, default=10, help='# of test images')
    parser.add_argument('--model_path', type=str, default=None, help='Test model path')
    parser.add_argument('--img_path', type=str, default=None, help='Image path to test')
    parser.add_argument('--deterministic_inv', type=int, default=1, help='Whether to use deterministic inversion during inference')
    parser.add_argument('--hybrid_noise', type=int, default=0, help='Whether to change multiple attributes by mixing multiple models')
    parser.add_argument('--model_ratio', type=float, default=1, help='Degree of change, noise ratio from original and finetuned model.')


    # Loss & Optimization
    parser.add_argument('--cgl_dcl_loss_w', type=int, default=3, help='Weights of CGL+ DCL loss')
    parser.add_argument('--l1_loss_w', type=float, default=0, help='Weights of L1 loss')
    parser.add_argument('--sa_loss_w', type=float, default=0, help='Weights of SA loss')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/16', help='ViT-B/16, ViT-B/32, RN50x16 etc')
    parser.add_argument('--lr_clip_finetune', type=float, default=2e-6, help='Initial learning rate for finetuning')
    parser.add_argument('--lr_clip_lat_opt', type=float, default=2e-2, help='Initial learning rate for latent optim')
    parser.add_argument('--n_iter', type=int, default=1, help='# of iterations of a generative process with `n_train_img` images')
    parser.add_argument('--scheduler', type=int, default=1, help='Whether to increase the learning rate')
    parser.add_argument('--sch_gamma', type=float, default=1.3, help='Scheduler gamma')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if args.clip_finetune:
        if args.edit_attr is not None:
            args.exp = args.exp + f'_FT_{new_config.data.category}_{args.edit_attr}_t{args.t_0}_ninv{args.n_inv_step}_ngen{args.n_train_step}_id{args.sa_loss_w}_l1{args.l1_loss_w}_lr{args.lr_clip_finetune}'
        else:
            args.exp = args.exp + f'_FT_{new_config.data.category}_{args.trg_txts}_t{args.t_0}_ninv{args.n_inv_step}_ngen{args.n_train_step}_id{args.sa_loss_w}_l1{args.l1_loss_w}_lr{args.lr_clip_finetune}'

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(args.exp, exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    os.makedirs('precomputed', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs(args.exp, exist_ok=True)
    args.image_folder = os.path.join(args.exp, 'image_samples')
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            # shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder, exist_ok=True)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print("<" * 80)

    runner = Generation(args, config)
    try:
        if args.clip_finetune:
            runner.clip_finetune()
        else:
            print('Choose --clip_finetune')
            raise ValueError
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())