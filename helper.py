import torch 
import logging
def calculate_accuracy_prototypes(y_pred, y_train, plabels):
    with torch.no_grad():
        idx = torch.argmin(y_pred, axis=1)
        return torch.true_divide(torch.sum(y_train == plabels[idx]), len(y_pred)) * 100

def predict_label(y_pred,plabels):
    with torch.no_grad():
        return plabels[torch.argmin(y_pred,1)]

def setup_logger(args,name="datl"):
    level    = logging.INFO
    format   = '  %(message)s'
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]] 
    dataset = str(args.source_dir.split("/")[-2])+"_"+str(args.target_dir.split("/")[-2])
    handlers = [logging.FileHandler(args.log_path+"log_"+name+"_"+dataset+".txt"), logging.StreamHandler()]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    return logging