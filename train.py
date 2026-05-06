import os, torch, torch.nn as nn, torch.optim as optim, argparse, logging
from collections import OrderedDict
import data as Data, utils.logger as Logger, utils.metrics as Metrics
from utils.metric_tools import ConfuseMatrixMeter
from utils.loss import ce_love
from models.model import GCNMamba
from utils.torchutils import get_scheduler, save_network

def create_CD_model(opt):
    m = opt['model']
    if m['name'] == 'gcnmamba':
        return GCNMamba(out_channels=m['n_classes'], conv_mode=m.get('conv_mode', 'deepwise'),
            stage=m.get('stage', 4), mamba_act=m.get('mamba_act', 'silu'),
        )
    raise NotImplementedError()

def run_step(model, loader, device, loss_fn, metric, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    metric.clear()
    for i, data in enumerate(loader):
        a, b, gt = data['A'].to(device), data['B'].to(device), data['L'].to(device).long()
        with torch.set_grad_enabled(is_train):
            pred = model(a, b)
            loss = loss_fn(pred, gt)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        out = torch.argmax(pred.detach(), dim=1)
        res = metric.update_cm(pr=out.cpu().numpy(), gt=gt.cpu().numpy())
        if i % 10 == 0: print(f"{'Train' if is_train else 'Val'} Step {i}: mF1 {res.item():.4f}")
    return metric.get_scores()['mf1']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config/config.yaml')
    p.add_argument('--phase', type=str, default='train')
    p.add_argument('--gpu_ids', type=str, default='0')
    opt = Logger.dict_to_nonedict(Logger.parse(p.parse_args()))
    
    Logger.setup_logger(None, opt['path_cd']['log'], 'train', logging.INFO, True)
    logger = logging.getLogger('base')
    
    dev = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    loaders = {p: Data.create_cd_dataloader(Data.create_cd_dataset(opt['datasets'][p], p), opt['datasets'][p], p) for p in opt['datasets']}
    
    model = create_CD_model(opt).to(dev)
    if opt['gpu_ids'] and len(opt['gpu_ids']) > 1: model = nn.DataParallel(model)
    
    loss_fn = ce_love  if opt['model']['loss'] == 'ce_love' else nn.CrossEntropyLoss()
    cfg_opt = opt['train']['optimizer']
    if cfg_opt['type'] == 'adam': optm = optim.Adam(model.parameters(), lr=cfg_opt['lr'])
    elif cfg_opt['type'] == 'adamw': optm = optim.AdamW(model.parameters(), lr=cfg_opt['lr'])
    else: optm = optim.SGD(model.parameters(), lr=cfg_opt['lr'], momentum=0.9)
    
    metric, best_f1 = ConfuseMatrixMeter(n_class=2), 0.0
    for epoch in range(opt['train']['n_epoch']):
        logger.info(f"Epoch {epoch} LR: {optm.param_groups[0]['lr']:.7f}")
        run_step(model, loaders['train'], dev, loss_fn, metric, optm)
        if epoch % opt['train']['val_freq'] == 0:
            val_f1 = run_step(model, loaders['val'], dev, loss_fn, metric)
            is_best = val_f1 > best_f1
            if is_best: best_f1 = val_f1
            save_network(opt, epoch, model, optm, is_best)
            logger.info(f"Summary: Val mF1 {val_f1:.4f} (Best: {best_f1:.4f})")
        get_scheduler(optm, opt['train']).step()

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()