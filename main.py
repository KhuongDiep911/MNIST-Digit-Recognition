import os
import tqdm
import random
import copy
import shutil
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
# from networks.cnn import CNN
from tensorboardX import SummaryWriter
from config.cfg import parse
from networks.build import build_model, build_optimizer, build_scheduler
import matplotlib.pyplot as plt
# from train import train

def train(model, loader, cfg, device):
    # Option
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_func = torch.nn.CrossEntropyLoss()

    # Log
    if os.path.exists(cfg.log_path):
        shutil.rmtree(cfg.log_path)
    writer = SummaryWriter(cfg.log_path)

    # Train
    step = 1
    best_accuracy = 0.0
    best_state_dict = None
    #loss_keeper={'train':[]}###################
    accuracy_keeper={'train':[], 'valid':[]}############
    loss_keeper={'train':[], 'valid':[]}
    for epoch in range(1, cfg.num_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0 #########
        valid_loss = 0.0 #########
        tl=0########

        for images, labels in tqdm.tqdm(loader['train'], desc='train: '):
            step_ = step * cfg.train_batch_size
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = torch.max(outputs, dim=-1)[1]
            tl += (preds == labels).sum().item()
            train_loss+=loss.item()

            if step % cfg.print_freq == 0:
                lr = scheduler.get_last_lr()[0]
                print(f'epoch: {epoch}/{cfg.num_epochs} | loss: {loss.item():6f} | lr: {lr:e}')
                writer.add_scalar('loss', loss, step_) ######
                writer.add_scalar('lr', lr, step_)
            step += 1

        # Val
        tp = 0

        model.eval()
        for images, labels in tqdm.tqdm(loader['val'], desc='val: '):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)############
            preds = torch.max(outputs, dim=-1)[1]
            tp += (preds == labels).sum().item()
            valid_loss+=loss.item()############

        train_accuracy = float(tl) / float(len(loader['train'].dataset))########## 
        train_loss = float(train_loss) / float(len(loader['train'].dataset))########

        valid_accuracy = float(tp) / float(len(loader['val'].dataset)) #1-valid_loss
        valid_loss = float(valid_loss) / float(len(loader['val'].dataset))##########
        
        accuracy = float(tp) / float(len(loader['val'].dataset))

        print(f'epoch: {epoch}/{cfg.num_epochs} | accuracy: {accuracy:.3f}')########3
        # writer.add_scalar('loss', valid_loss, step_)#########
        
        writer.add_scalar('accuracy', accuracy, step_)
        accuracy_keeper['train'].append(train_accuracy)###############
        accuracy_keeper['valid'].append(valid_accuracy)
        loss_keeper['train'].append(train_loss)
        loss_keeper['valid'].append(valid_loss)
        

        if accuracy > best_accuracy:########
            best_accuracy = accuracy
            best_state_dict = copy.deepcopy(model.state_dict())
        print(f'best accuracy: {best_accuracy:.3f}')

        scheduler.step()
    writer.close()

    # Save best model
    model_filename = os.path.join(cfg.model_path, cfg.model_name + '.pkl')
    torch.save(best_state_dict, model_filename)

    return loss_keeper, accuracy_keeper


if __name__ == '__main__':
    # Parameter
    cfg = parse()
    # print(cfg)
    # print("=====", cfg.CNN.num_classes)
    os.makedirs(cfg.model_path, exist_ok=True)

    # Use GPU or CPU
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda' if use_gpu else 'cpu')
    # print('use_gpu: ', device)

    # Seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    if use_gpu:
        torch.cuda.manual_seed_all(cfg.seed)

    # Load model
    model = build_model(cfg).to(device)
    # model = CNN().to(device)
    summary(model, (1,28,28))

    # Load dataset
    train_dataset = datasets.MNIST(root=cfg.dataset_path, train=True, transform=transforms.ToTensor(), download=cfg.download)
    val_dataset = datasets.MNIST(root=cfg.dataset_path, train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.train_batch_size,
                                               num_workers=cfg.num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=cfg.test_batch_size,
                                               num_workers=cfg.num_workers, shuffle=False)
    loader = {'train': train_loader, 'val': val_loader}

    # Train network
    loss_keeper, accuracy_keeper=train(model, loader, cfg, device)


    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(accuracy_keeper['train'],label="Training Accuracy")
    plt.plot(accuracy_keeper['valid'],label="Validation Accuracy")
    plt.ylim(0.93, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(loss_keeper['train'],label="Training Loss")
    plt.plot(loss_keeper['valid'],label="Validation Loss")
    plt.ylim(0.0225, 0.0245)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.suptitle(cfg.title)
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.show()
    