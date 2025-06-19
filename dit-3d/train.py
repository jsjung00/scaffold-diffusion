import sys 
from pathlib import Path 
import os 
import wandb 

sys.path.append(str(Path(__file__).parent.parent))
import dataloader

class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type:str):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type)

        if args.window_size > 0:
            self.model = DiT3D_models_WindAttn[args.model_type](pretrained=args.use_pretrained, 
                                                   input_size=args.voxel_size, 
                                                   window_size=args.window_size, 
                                                   window_block_indexes=args.window_block_indexes, 
                                                   num_classes=args.num_classes
                                                )
        else:
            self.model = DiT3D_models[args.model_type](pretrained=args.use_pretrained, 
                                                    input_size=args.voxel_size, 
                                                    num_classes=args.num_classes
                                                    )


    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, y, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0, y, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t, y):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t, y)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data):
        B, D, N = data.shape                           # [16, 3, 2048]
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        noises = torch.randn_like(data)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, y=y)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, shape, device, y, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, y, freq, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, y=y, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas

def train(gpu, opt, output_dir, noises_init):
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)
    
    if opt.use_ema:
        ema = deepcopy(model).to("cuda")
        requires_grad(ema, False)
    
    model = model.cuda()

    print("Model = %s" % str(model))
    total_params = sum(param.numel() for param in model.parameters())/1e6
    print("Total_params = %s MB " % str(total_params))   

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0)

    if opt.use_ema:
        update_ema(ema, model, decay=0)
        model.train()
        ema.eval()

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)
    
    def new_y_chain(y, num_chain, num_classes):
        return torch.randint(low=0, high=num_classes, size=(num_chain,), device=y.device)
    

    start_epoch = 0 
    for epoch in range(start_epoch, opt.niter):
        for i, x in enumerate(dataloader):
            loss = model.get_loss_iter(x).mean()
            optimizer.zero_grad()
            loss.backward()

            if opt.grad_clip is not None:
                torch.nn.utis.clip_grad_norm_(model.parameters(), opt.grad_clip)
            
            optimizer.step()

            if opt.use_ema:
                update_ema(ema, model)

            # logging 
            global_step = i + len(dataloader)*epoch

            wandb.log({"train/loss": loss.item(), "train_lr": optimizer.param_groups[0]['lr']}, step=global_step)

            # TODO: add visualization  

        if (epoch + 1) % opt.saveIter == 0:
            save_dict = {'epoch': epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
            if opt.use_ema:
                save_dict.update({'ema': ema.state_dict()})
            torch.save(save_dict, f"{output_dir}/epoch_{epoch}.pth")


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main():
    opt = parse_args()
    output_dir = '/home/jsjung00/Desktop/Code/voxeldiffusion/dit-3d/output'
    tokenizer = MinecraftTokenizer(config, config.data.air_not_air) # can later play around with changing mask token 

    # TODO: need to change the dataloader shape... because I think it expects (B, C, N)
    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    noises_init = torch.randn(len(train_ds), config.data.voxel_side_len**3, opt.nc)
    
    train(opt, output_dir, noises_init)



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='dit3d', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--dataroot', default='ShapeNetCore.v2.PC15k/')
    parser.add_argument('--category', default='chair')
    parser.add_argument('--num_classes', type=int, default=55)

    parser.add_argument('--bs', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument("--voxel_size", type=int, choices=[16, 32, 64, 128, 256], default=64)
    
    '''model'''
    parser.add_argument("--model_type", type=str, choices=list(DiT3D_models.keys()), default="DiT-XL/2")
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', type=int, default=1000)

    #params
    parser.add_argument('--window_size', type=int, default=0)
    parser.add_argument('--window_block_indexes', type=tuple, default='0,3,6,9')
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='', help="path to model (to continue training)")


    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveIter', default=100, type=int, help='unit: epoch')
    parser.add_argument('--diagIter', default=50000, type=int, help='unit: epoch')
    parser.add_argument('--vizIter', default=50000, type=int, help='unit: epoch')
    parser.add_argument('--print_freq', default=50, type=int, help='unit: iter')

    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    parser.add_argument('--debug', action='store_true', default=False, help = 'debug mode')
    parser.add_argument('--use_tb', action='store_true', default=False, help = 'use tensorboard')
    parser.add_argument('--use_pretrained', action='store_true', default=False, help = 'use pretrained 2d DiT weights')
    parser.add_argument('--use_ema', action='store_true', default=False, help = 'use ema')

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()