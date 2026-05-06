import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import logging
import os
logger = logging.getLogger('base')

def get_scheduler(optimizer, args):
    if args['sheduler']['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args['n_epoch'] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args['sheduler']['lr_policy'] == 'step':
        step_size = args['n_epoch']//args['sheduler']['n_steps']
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args['sheduler']['gamma'])
    elif args['sheduler']['lr_policy'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args['n_epoch'],
            eta_min=args['sheduler']['min_lr'] if 'min_lr' in args['sheduler'] else 0
        )
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented' % args['sheduler']['lr_policy'])
    return scheduler

def save_network(opt, epoch, cd_model, optimizer, is_best_model=False, best_mF1=0.0):
    cd_gen_path = os.path.join(
        opt['path_cd']['checkpoint'], 'cd_model_E{}_gen.pth'.format(epoch))
    cd_opt_path = os.path.join(
        opt['path_cd']['checkpoint'], 'cd_model_E{}_opt.pth'.format(epoch))
        
    network = cd_model
    if isinstance(cd_model, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    # torch.save(state_dict, cd_gen_path)  # If you want to only save best
    
    if is_best_model:
        best_cd_gen_path = os.path.join(
            opt['path_cd']['checkpoint'], 'best_cd_model_gen.pth')
        best_cd_opt_path = os.path.join(
            opt['path_cd']['checkpoint'], 'best_cd_model_opt.pth')
        torch.save(state_dict, best_cd_gen_path)

    opt_state = {'epoch': epoch,
                 'best_mF1': best_mF1,
                 'optimizer': optimizer.state_dict()}
    
    if is_best_model:
        torch.save(opt_state, best_cd_opt_path)

    if is_best_model:
        logger.info('Saved best CD model in [{:s}] ...'.format(best_cd_gen_path))
