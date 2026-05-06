from collections import OrderedDict
import os, torch, torch.nn as nn, argparse, logging
import data as Data, utils.logger as Logger, utils.metrics as Metrics
from utils.metric_tools import ConfuseMatrixMeter
from models.model import GCNMamba

def create_CD_model(opt):
    m = opt['model']
    if m['name'] == 'gcnmamba':
        return GCNMamba(out_channels=m['n_classes'], conv_mode=m.get('conv_mode', 'deepwise'),
            stage=m.get('stage', 4), mamba_act=m.get('mamba_act', 'silu'),
        )
    raise NotImplementedError()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config/config.yaml')
    p.add_argument('--phase', type=str, default='test')
    p.add_argument('--gpu_ids', type=str, default=None)
    args = p.parse_args()
    opt = Logger.dict_to_nonedict(Logger.parse(args))
    
    Logger.setup_logger(None, opt['path_cd']['log'], 'test', logging.INFO, True)
    logger, logger_test = logging.getLogger('base'), logging.getLogger('test')
    
    dev = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    ds_opt = opt['datasets']['test']
    test_loader = Data.create_cd_dataloader(Data.create_cd_dataset(ds_opt, 'test'), ds_opt, 'test')
    
    model = create_CD_model(opt)
    load_path = opt["path_cd"]["resume_state"]
    if load_path:
        logger.info(f"Loading model from {load_path}_gen.pth")
        model.load_state_dict(torch.load(f"{load_path}_gen.pth", map_location='cpu'), strict=False)
    
    model = model.to(dev)
    if opt['gpu_ids'] and len(opt['gpu_ids']) > 1: model = nn.DataParallel(model)
    
    model.eval()
    metric = ConfuseMatrixMeter(n_class=2)
    res_path = os.path.join(opt['path_cd']['result'], 'test')
    os.makedirs(res_path, exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            a, b, gt = data['A'].to(dev), data['B'].to(dev), data['L'].to(dev).long()
            pred = model(a, b)
            out = torch.argmax(pred.detach(), dim=1)
            f1 = metric.update_cm(pr=out.cpu().numpy(), gt=gt.cpu().numpy())
            
            logger_test.info(f"Iter {i}/{len(test_loader)}, running_mf1: {f1.item():.5f}")
            
            p_vis = (out.float().unsqueeze(1).repeat(1, 3, 1, 1) * 2.0 - 1.0)
            g_vis = (gt.float().unsqueeze(1).repeat(1, 3, 1, 1) * 2.0 - 1.0)
            grid = Metrics.tensor2img(torch.cat((a, b, p_vis, g_vis), dim=0))
            Metrics.save_img(grid, os.path.join(res_path, f"res_{i}.png"))

    scores = metric.get_scores()
    msg = "".join([f"{k}: {v:.4e} \n" for k, v in scores.items()])
    logger_test.info(f"Final Test mF1: {scores['mf1']:.5f}\n{msg}")
    logger.info("Test Finished.")

if __name__ == '__main__':
    main()