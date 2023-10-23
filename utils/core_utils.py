import numpy as np
import torch
from utils.utils import *
from utils.loss_func import NLLSurvLoss, CrossEntropySurvLoss, MultiSurvLoss

import os
from datasets.dataset_generic import split_slideinfo
from models.model_MIL import MIL_fc, MIL_fc_mc
from models.model_CLAM import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sksurv.metrics import concordance_index_censored


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    if args.survival:
        train_split, val_split = datasets
    else:
        train_split, val_split, test_split = datasets
        print("Testing on {} samples".format(len(test_split)))
    
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    if args.survival: # TODO 指定survival 参数需指定alpha_surv和bag_loss in ce_surv or nll_surv
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        else:
            raise NotImplementedError

        reg_fn = l1_reg_all # 正则损失

    else:
        if args.bag_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
            if device.type == 'cuda':
                loss_fn = loss_fn.cuda()
        else:
            loss_fn = nn.CrossEntropyLoss()


    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.survival:
            model_dict.update({'surv': args.survival})
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, surv=args.survival)
    val_loader = get_split_loader(val_split,  testing = args.testing, surv=args.survival)
    if not args.survival:
        test_loader = get_split_loader(test_split, testing = args.testing)

    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    
    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster and not args.survival:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        elif args.survival:
            train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
            stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir)
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    if args.survival:
        results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes)

        print('Val c-Index: {:.4f}'.format(val_cindex))
        writer.close()
        return results_val_dict, 0, val_cindex, 0, 0

    else:
        _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

        if writer:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_auc', val_auc, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)
            writer.close()
        return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_survival(epoch, model, loader, optimizer, scheduler, writer = None, loss_fn = None,  reg_fn=None, lambda_reg=0.0, gc=16):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    optimizer.zero_grad() # 最好先在一个epoch的所有batch开始之前进行一次zero grad清零
    for batch_idx, (data, _, y_disc, event_time, censor) in enumerate(loader):
        data = data.to(device).type(torch.float32)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)

        output = model(data)

        if not isinstance(output, tuple):
            loss = loss_fn(h=output, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
        else:
            logits_hazard, Y_prob, Y_hat, attention_raw, results_dict = output
            loss = 0.5*loss_fn(h=logits_hazard, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            # output = logits_hazard

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        if isinstance(loss_fn, (NLLSurvLoss, CrossEntropySurvLoss, MultiSurvLoss)):
            hazards = torch.sigmoid(output)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = output.detach().cpu().numpy().squeeze()

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if y_disc.shape[0] == 1 and (batch_idx + 1) % 5 == 0:
            print('batch {}, loss_surv: {:.4f}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value, loss_value + loss_reg, y_disc.detach().cpu().item(), float(event_time.detach().cpu().item()), float(risk), data.size(0)))
        elif y_disc.shape[0] != 1 and (batch_idx + 1) % 5 == 0:
            print('batch {}, loss_surv: {:.4f}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value, loss_value + loss_reg, y_disc.detach().cpu()[0], float(event_time.detach().cpu()[0]), float(risk[0]), data.size(0)))

        # backward pass
        loss = loss / gc + loss_reg # 这里做法确定没问题 https://zhuanlan.zhihu.com/p/595716023 累积gc次进行参数更新导致梯度值也被放大gc倍，所以要/gc
        loss.backward() # 仅计算梯度，周期叠加，当满足gc条件才进行一个参数更新

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step() # TODO 这里还没确定，目前是在epoch层面更新learningrate；考虑在iteration层面更新，那对应到scheduler初始化那里的值也要调整
    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, scheduler, n_classes, writer = None, loss_fn = None, reg_fn=None, lambda_reg=0.0, gc=16, **kwargs):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    optimizer.zero_grad() # 最好先在一个epoch的所有batch开始之前进行一次zero grad清零

    for batch_idx, (data, label, slide_id) in enumerate(loader):
        data = data.to(device).type(torch.float32)
        label = label.to(device)

        logits, Y_prob, Y_hat, _, results_dict = model(data, label=label, prototype=kwargs["prototypes"])
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss += results_dict["instance_loss"] if "instance_loss" in results_dict else 0.

        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 10 == 0:
            print('batch {}, CE loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss = loss / gc + loss_reg # 这里做法确定没问题 https://zhuanlan.zhihu.com/p/595716023 累积gc次进行参数更新导致梯度值也被放大gc倍，所以要/gc
        loss.backward() # 仅计算梯度，周期叠加，当满足gc条件才进行一个参数更新

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step() # TODO 这里还没确定，目前是在epoch层面更新learningrate；考虑在iteration层面更新，那对应到scheduler初始化那里的值也要调整

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_total_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None, **kwargs):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, _) in enumerate(loader):
            data, label = data.to(device, non_blocking=True).type(torch.float32), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, results_dict = model(data, label=label, prototype=kwargs["prototypes"])

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)
            loss += results_dict["instance_loss"] if "instance_loss" in results_dict else 0.

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    with torch.no_grad():
        for batch_idx, (data, _, y_disc, event_time, censor) in enumerate(loader):
            data = data.to(device).type(torch.float32)
            y_disc = y_disc.to(device)
            event_time = event_time.to(device)
            censor = censor.to(device)

            output = model(data)

            if not isinstance(output, tuple):
                loss = loss_fn(h=output, y=y_disc, t=event_time, c=censor)
                loss_value = loss.item()
            else:
                logits_hazard, Y_prob, Y_hat, attention_raw, results_dict = output
                loss = 0.5*loss_fn(h=logits_hazard, y=y_disc, t=event_time, c=censor)
                loss_value = loss.item()
                # output = logits_hazard

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg

            if isinstance(loss_fn, (NLLSurvLoss, CrossEntropySurvLoss, MultiSurvLoss)):
                hazards = torch.sigmoid(output)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            else:
                risk = output.detach().cpu().numpy().squeeze()

            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = censor.detach().cpu().numpy()
            all_event_times[batch_idx] = event_time.detach().cpu().numpy()

            val_loss_surv += loss_value
            val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss_surv, val_loss, c_index))

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_best_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes, **kwargs):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = []

    for batch_idx, (data, label, _) in enumerate(loader):
        data, label = data.to(device).type(torch.float32), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data, prototype=kwargs["prototypes"])

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        probs = probs.squeeze(0)
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.append({'slide_id': np.array(slide_id), 'prob_neg': probs[0], 'prob_pos': probs[1], 'label': label.item()})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger


def summary_survival(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = []

    for batch_idx, (data, _, y_disc, event_time, censor) in enumerate(loader):
        data = data.to(device).type(torch.float32)

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            output = model(data)
    
        if isinstance(output, tuple):
            output = output[2]

        if output.shape[1] > 1:
            hazards = torch.sigmoid(output)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = output.detach().cpu().numpy().squeeze()

        event_time = event_time.numpy()
        censor = censor.numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor
        all_event_times[batch_idx] = event_time
        patient_results.append({'slide_id': slide_id, 'risk': risk[0], 'disc_label': y_disc.item(), 'survival': event_time[0], 'censorship': censor[0]})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index