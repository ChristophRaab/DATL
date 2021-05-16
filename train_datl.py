from __future__ import print_function, division
import sys,os
import os.path as path
from torchvision.datasets.folder import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader
from kmeans_pytorch import kmeans
import numpy as np
from itertools import cycle
import time
from loss import entropy,mpce
from make_config import make_parser
from parameter_init_adjustments import init_weights,cdann_lda_coeff,inv_lr_scheduler
from helper import setup_logger
from network import create_resnet50_features,grl_hook,SiameseTangentLayer
from transformations import training_augmentation,validation_augmentation



def load_data(loader,featuer_extractor,args):
    d = torch.empty([0,args.bottleneck_dim]).to(args.cuda)
    p = torch.empty(0,3,224,224).to(args.cuda)
    y = torch.empty([0]).to(args.cuda)
    for x,yi in loader:
        x,yi = x.to(args.cuda),yi.to(args.cuda)
        d,y,p = torch.cat([d,featuer_extractor(x)]),torch.cat([y,yi]),torch.cat([p,x])

    return [d,y],p

def train_datl_sia(args):
    start = time.time()
    name ="datl_sia_mpce"
    logging = setup_logger(args,name=name)
    logging.info(str(args))

    source_dataset = ImageFolder(args.source_dir,transform=training_augmentation())
    target_dataset = ImageFolder(args.target_dir,transform=training_augmentation())
    validation_dataset = ImageFolder(args.target_dir,transform=validation_augmentation())
    source_loader = DataLoader(source_dataset,shuffle=True,num_workers=args.num_workers,batch_size=args.batch_size,drop_last=True,pin_memory=True)
    target_loader = DataLoader(target_dataset,shuffle=True,num_workers=args.num_workers,batch_size=args.batch_size,drop_last=True,pin_memory=True)
    validation_loader = DataLoader(validation_dataset,shuffle=False,num_workers=args.num_workers,batch_size=args.batch_size,pin_memory=True)

    source_init_dataset = ImageFolder(args.source_dir,transform=validation_augmentation())
    source_init_loader = DataLoader(source_init_dataset,shuffle=False,num_workers=args.num_workers,batch_size=args.batch_size)

    source_loader_size,target_loader_size,validation_loader_size = len(source_loader),len(target_loader),len(validation_loader)
    source_dataset_size, target_dataset_size = len(source_dataset),len(target_dataset)
    num_classes = len(source_dataset.classes)
    

    features_extractor = nn.Sequential(create_resnet50_features(),nn.Flatten(),nn.Linear(2048,args.bottleneck_dim),nn.LeakyReLU(),nn.LayerNorm(args.bottleneck_dim)).to(args.cuda)
    classifier =nn.Sequential(nn.Linear(args.bottleneck_dim,num_classes)).to(args.cuda)
    classifier.apply(init_weights),features_extractor[-1].apply(init_weights)
    features_extractor,classifier= features_extractor.to(args.cuda),classifier.to(args.cuda)

    with torch.no_grad():
        sd, sp = load_data(source_init_loader,features_extractor,args)
        td, tp = load_data(validation_loader,features_extractor,args)
        td[-1] = None
        source_siaprotos = torch.empty(0,3,224,224).to(args.cuda)
        target_siaprotos = torch.empty(0,3,224,224).to(args.cuda)
        for y in torch.unique(sd[-1]):
            class_img = sp[sd[-1] == y,:]
            class_sia = class_img[torch.randint(class_img.size(0),(1,))]
            source_siaprotos = torch.cat([source_siaprotos,class_sia])
        yt,_ =  kmeans(td[0],num_classes,device=td[0].device)
        for y in torch.unique(yt):
            class_img = tp[yt == y,:]
            class_sia = class_img[torch.randint(class_img.size(0),(1,))]
            target_siaprotos = torch.cat([target_siaprotos,class_sia])
        siaprotos = torch.cat([source_siaprotos,target_siaprotos])

    discriminator = SiameseTangentLayer(2,args.num_protos,args.bottleneck_dim,args.subspace_dim)
    discriminator = discriminator.to(args.cuda)
    discriminator.init(siaprotos,[sd,td])
    sd = td = siaprotos = source_siaprotos = target_siaprotos = source_init_dataset = source_init_loader = 0
    torch.cuda.empty_cache()

    optimizer = optim.SGD(
        [{'params': features_extractor[:-1].parameters(),"lr_mult":1,'decay_mult':2},
        {'params': features_extractor[-1].parameters(),"lr_mult":10,'decay_mult':2},
        {'params': classifier.parameters(),"lr_mult":10,'decay_mult':2}],
        lr=args.lr,nesterov=True,momentum=0.9,weight_decay=0.0005)
    disop = optim.Adam([{'params': discriminator.parameters(),"lr_mult":10,'decay_mult':2}], lr=args.dlr,weight_decay=0.0005)# 0.005 92.0 ohne disop scheduing

    best_acc, best_model = 0,copy.deepcopy([features_extractor,classifier,discriminator])   
    j=0
    for i in range(args.num_epochs):
        with torch.set_grad_enabled(True):
            avg_loss = avg_acc = avg_dc  = classifier_loss = discriminator_loss = loss = 0.0
            training_list = zip(source_loader, cycle(target_loader)) if len(source_loader) > len(target_loader) else zip(cycle(source_loader), target_loader)
            for (xs,ys),(xt,yt) in training_list:
                
                xs,ys,xt,yt = xs.to(args.cuda),ys.to(args.cuda),xt.to(args.cuda),yt.to(args.cuda)
                features_extractor.train(),classifier.train(),discriminator.train(),
                optimizer = inv_lr_scheduler(optimizer,j,gamma=0.001,power=0.75)
                disop = inv_lr_scheduler(disop,j,gamma=0.001,power=0.75)
                optimizer.zero_grad(),disop.zero_grad()
            
                fes =features_extractor(xs)
                fet = features_extractor(xt)

                ls = classifier(fes)
                lt = classifier(fet)
                
                classifier_loss = nn.CrossEntropyLoss()(ls,ys)
                entropy_loss = entropy(lt)

                yd = torch.from_numpy(np.array([0] * fes.size(0) + [1] * fet.size(0))).long().to(args.cuda)     
                fe = torch.cat([fes,fet],dim=0)

                protos = features_extractor(discriminator.protos)
                yp = torch.from_numpy(np.array([0] * int(protos.size(0)/2) + [1] * int(protos.size(0)/2))).long().to(args.cuda)    
                # protos = protos.detach()
                
                j += 1
                lda = cdann_lda_coeff(j)
                fe.register_hook(grl_hook(lda))
                dis = discriminator(fe,protos)    
                discriminator_loss =  mpce(dis,yd,yp,invert_distance=args.invert)

                loss = classifier_loss + discriminator_loss +args.lent * entropy_loss
                loss.backward()
                optimizer.step(),disop.step()

                discriminator.orthogonalize_subspace()
                _,preds = nn.Softmax(1)(ls).detach().max(1)
                
                avg_loss = avg_loss + loss
                avg_dc = avg_dc + discriminator_loss
                avg_acc  = avg_acc + (preds == ys).sum()

        if i % args.eval_epoch == 0:
            with torch.set_grad_enabled(False):

                vavg_loss,vavg_acc = 0.0,0.0
                for xt,yt in validation_loader:
                    
                    xt,yt = xt.to(args.cuda),yt.to(args.cuda)
                    features_extractor.eval(),classifier.eval(),discriminator.eval()

                    lt = classifier(features_extractor(xt))

                    _,preds = nn.Softmax(1)(lt).max(1)
                    classifier_loss = nn.CrossEntropyLoss()(lt,yt)
                    
                    loss = classifier_loss

                    vavg_loss = vavg_loss + loss
                    vavg_acc  = vavg_acc + (preds == yt).sum()

                vavg_acc = (vavg_acc/target_dataset_size).item()
                if best_acc < vavg_acc:
                    best_acc,best_model= vavg_acc, copy.deepcopy([features_extractor,classifier,discriminator])
                logging.info("Progress " + str(i) + ", " +str(j) + ", Mean Validation Loss: "+str(round((vavg_loss/validation_loader_size).item(),3))+ ", Acc :"+str(round(vavg_acc,3))
                    + " --- Mean Training Loss: "+str(round((avg_loss/source_loader_size).item(),3))+ ", Acc :"+str(round((avg_acc/source_dataset_size).item(),3)) + ", DC :"+ str(round((avg_dc/(source_loader_size + target_loader_size)).item(),2)))
    
    torch.save(best_model[0], args.model_path+"best_"+name+"_fe_"+str(args.source_dir.split("/")[-2])+"_"+str(args.target_dir.split("/")[-2])+".pth.tar")
    torch.save(best_model[1], args.model_path+"best_"+name+"_classifier_"+str(args.source_dir.split("/")[-2])+"_"+str(args.target_dir.split("/")[-2])+".pth.tar")
    torch.save(best_model[-1], args.model_path+"best_"+name+"_tangent_"+str(args.source_dir.split("/")[-2])+"_"+str(args.target_dir.split("/")[-2])+".pth.tar")
    duration = round((time.time() - start) / 60,2)
    logging.info("Finished in "+str(duration) +" minutes with Best Acc" +str(best_acc)+"========================================================================================")
    return best_acc,best_model

if __name__ == "__main__":

    args = make_parser()
    train_datl_sia(args)
