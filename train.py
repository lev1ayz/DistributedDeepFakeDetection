import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import models
from utils.logger import Logger
import myexman
from utils import utils
import sys
import torch.multiprocessing as mp
import torch.distributed as dist
import socket
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def add_learner_params(parser):
    parser.add_argument('--problem', default='sim-clr',
        help='The problem to train',
        choices=models.REGISTERED_MODELS,
    )
    parser.add_argument('--name', default='',
        help='Name for the experiment',
    )
    parser.add_argument('--ckpt', default='',
        help='Optional checkpoint to init the model.'
    )
    parser.add_argument('--verbose', default=False, type=bool)
    # optimizer params
    parser.add_argument('--lr_schedule', default='warmup-anneal')
    parser.add_argument('--opt', default='lars', help='Optimizer to use', choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--iters', default=-1, type=int, help='The number of optimizer updates')
    parser.add_argument('--warmup', default=0, type=float, help='The number of warmup iterations in proportion to \'iters\'')
    parser.add_argument('--lr', default=0.1, type=float, help='Base learning rate')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, dest='weight_decay')
    # trainer params
    parser.add_argument('--save_freq', default=10000000000000000, type=int, help='Frequency to save the model')
    parser.add_argument('--log_freq', default=100, type=int, help='Logging frequency')
    parser.add_argument('--eval_freq', default=10000000000000000, type=int, help='Evaluation frequency')
    parser.add_argument('-j', '--workers', default=4, type=int, help='The number of data loader workers')
    parser.add_argument('--eval_only', default=False, type=bool, help='Skips the training step if True')
    parser.add_argument('--seed', default=-1, type=int, help='Random seed')
    parser.add_argument('--root', default='/mnt/results', type=str, help='Root')
    parser.add_argument('--deepfakes', default=False, type=bool, help='load deepfakes')
    parser.add_argument('--pretrained', default=False, type=bool, help='load resnet with torchvision pretrained weights')
    parser.add_argument('--new_transforms', default=True, type=bool, help='use special FF transforms')

    # parallelizm params:
    parser.add_argument('--dist', default='dp', type=str,
        help='dp: DataParallel, ddp: DistributedDataParallel',
        choices=['dp', 'ddp'],
    )
    parser.add_argument('--dist_address', default='127.0.0.1:1234', type=str,
        help='the address and a port of the main node in the <address>:<port> format'
    )
    parser.add_argument('--node_rank', default=0, type=int,
        help='Rank of the node (script launched): 0 for the main node and 1,... for the others',
    )
    parser.add_argument('--world_size', default=1, type=int,
        help='the number of nodes (scripts launched)',
    )


def main():
    parser = myexman.ExParser(file=os.path.basename(__file__))
    add_learner_params(parser)

    is_help = False
    if '--help' in sys.argv or '-h' in sys.argv:
        sys.argv.pop(sys.argv.index('--help' if '--help' in sys.argv else '-h'))
        is_help = True

    args, _ = parser.parse_known_args(log_params=False)

    models.REGISTERED_MODELS[args.problem].add_model_hparams(parser)

    if is_help:
        sys.argv.append('--help')
    print('args:', args)
    args = parser.parse_args(namespace=args)
    print('args:', args)


    if args.data == 'imagenet' and args.aug == False:
        raise Exception('ImageNet models should be eval with aug=True!')

    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.gpu = 0
    ngpus = torch.cuda.device_count()
    args.number_of_processes = 1
    if args.dist == 'ddp':
        # add additional argument to be able to retrieve # of processes from logs
        # and don't change initial arguments to reproduce the experiment
        args.number_of_processes = args.world_size * ngpus
        parser.update_params_file(args)

        args.world_size *= ngpus
        mp.spawn(
            main_worker,
            nprocs=ngpus,
            args=(ngpus, args),
        )
    else:
        parser.update_params_file(args)
        main_worker(args.gpu, -1, args)


def main_worker(gpu, ngpus, args):
    fmt = {
        'train_time': '.3f',
        'val_time': '.3f',
        'lr': '.1e',
    }
    logger = Logger('logs', base=args.root, fmt=fmt)

    args.gpu = gpu
    torch.cuda.set_device(gpu)
    args.rank = args.node_rank * ngpus + gpu

    device = torch.device('cuda:%d' % args.gpu)

    if args.dist == 'ddp':
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://%s' % args.dist_address,
            world_size=args.world_size,
            rank=args.rank,
        )

        n_gpus_total = dist.get_world_size()
        assert args.batch_size % n_gpus_total == 0
        args.batch_size //= n_gpus_total
        if args.rank == 0:
            print(f'===> {n_gpus_total} GPUs total; batch_size={args.batch_size} per GPU')

        print(f'===> Proc {dist.get_rank()}/{dist.get_world_size()}@{socket.gethostname()}', flush=True)

    # create model
    model = models.REGISTERED_MODELS[args.problem](args, device=device)

    if args.ckpt != '':
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])

    # Data loading code
    model.prepare_data()
    train_loader, val_loader = model.dataloaders(iters=args.iters)

    # define optimizer
    cur_iter = 0
    optimizer, scheduler = models.ssl.configure_optimizers(args, model, cur_iter - 1)

    # optionally resume from a checkpoint
    if args.ckpt and not args.eval_only:
        optimizer.load_state_dict(ckpt['opt_state_dict'])

    cudnn.benchmark = True

    continue_training = args.iters != 0
    data_time, it_time = 0, 0

    while continue_training:
        train_logs = []
        model.train()

        start_time = time.time()
        for batch in tqdm(train_loader):
            cur_iter += 1
            batch = [item.to(device) for item in batch]
            #print('train.py batch len:', len(batch))
            # Plot images for testing
            """
            print('plotting a test pic in train.py')
            x, y, labels = batch
            print('labels:', labels)
            num_images = len(x) * 2
            assert len(x) == len(y), "images not same size?"
            print('num images:', num_images)
            fig, axs = plt.subplots(nrows=4, ncols=int(num_images/4))
            for i in range(0,num_images,2):
                img1 = x[int(i/2)]
                img1 = img1.cpu()
                img2 = y[int(i/2)]
                img2 = img2.cpu()
                #print('img shape:', img.shape)
                if(img1.shape[2] != 3): # if the channels are the 1st dim (0th dim), move them to last dim
                    img1 = torch.movedim(img1, 0, -1)
                if(img2.shape[2] != 3): # if the channels are the 1st dim (0th dim), move them to last dim
                    img2 = torch.movedim(img2, 0, -1)
                row = int(i/(num_images/4))
                col = int(i%(num_images/4))
                print('row:', row, ' col:', col)
                axs[row,col].imshow(img1)
                #axs[row, col+1].imshow(img2)
                #axs[row,col].title.set_text(labels[0][i])
            plt.show()
            plt.close('all')
            """
            # For FaceForensics, take both augmetations and merge them into 1 list
            if args.data == 'faceforensics':
                if args.problem == 'eval':
                    batch = [batch[0], batch[2]]
                else:
                    batch = [torch.cat((batch[0],batch[1])), batch[2]]
                """
                I don't know why batch[1] is also twice the specified batch size,
                I assume somewhere the code takes every item returned by the datasets __getitem()__
                augemnts it and then does another augmentation and adds it to the batch so that every positive
                example is at location i+original_batch_size where i is the current example so i discard batch[1].
                ITS BECAUSE OF THE MULTIPLIER!!!
                """


            data_time += time.time() - start_time

            logs = {}
            if not args.eval_only:
                # forward pass and compute loss
                logs = model.train_step(batch, cur_iter)
                loss = logs['loss']

                # gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # save logs for the batch
            train_logs.append({k: utils.tonp(v) for k, v in logs.items()})

            if cur_iter % args.save_freq == 0 and args.rank == 0:
                print('saving')
                save_checkpoint(args.root, model, optimizer, cur_iter)
            
            if cur_iter % args.eval_freq == 0 or cur_iter >= args.iters:
                # TODO: aggregate metrics over all processes
                print('aggregating metrics and evaluating')
                test_logs = []
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch = [x.to(device) for x in batch]
                        if args.data == 'faceforensics':
                            batch = [batch[0], batch[2]]
                        # forward pass
                        logs = model.test_step(batch)
                        # save logs for the batch
                        test_logs.append(logs)
                model.train()

                test_logs = utils.agg_all_metrics(test_logs)
                logger.add_logs(cur_iter, test_logs, pref='test_')
            it_time += time.time() - start_time

            if (cur_iter % args.log_freq == 0 or cur_iter >= args.iters) and args.rank == 0:
                #print('saving again?')
                #save_checkpoint(args.root, model, optimizer,cur_iter)
                train_logs = utils.agg_all_metrics(train_logs)

                logger.add_logs(cur_iter, train_logs, pref='train_')
                logger.add_scalar(cur_iter, 'lr', optimizer.param_groups[0]['lr'])
                logger.add_scalar(cur_iter, 'data_time', data_time)
                logger.add_scalar(cur_iter, 'it_time', it_time)
                logger.iter_info()
                logger.save()

                data_time, it_time = 0, 0
                train_logs = []

            if scheduler is not None:
                scheduler.step()

            if cur_iter >= args.iters:
                continue_training = False
                break

            start_time = time.time()
    if args.rank == 0:
        save_checkpoint(args.root, model, optimizer,cur_iter)

    if args.dist == 'ddp':
        dist.destroy_process_group()


def save_checkpoint(path, model, optimizer, cur_iter=None):
    if cur_iter is None:
        fname = os.path.join(path, 'checkpoint.pth.tar')
    else:
        fname = os.path.join(path, 'checkpoint-%d.pth.tar' % cur_iter)

    ckpt = model.get_ckpt()
    ckpt.update(
        {
            'opt_state_dict': optimizer.state_dict(),
            'iter': cur_iter,
        }
    )

    torch.save(ckpt, fname)


if __name__ == '__main__':
    main()
