from tqdm import tqdm
from typing import Tuple, List
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time

def trainTransformer(model: nn.Module, trainloader: DataLoader,
                    testloader: DataLoader, epochs: int = 25, lr_max: float = 0.01,
                    clip_norm: bool = True, scheduler: bool = False) -> Tuple[nn.Module, Tuple[List[float], List[float], List[float], List[float], List[float]]]:

    """Train a neural network

    Args:
        model (nn.Module): neural network to train
        trainloader (DataLoader): trainloader with train dataset
        testloader (DataLoader): testloader with test dataset
        epochs (int, optional): number of epoachs to train for. Defaults is 25.
        lr_max (float, optional): float  specifying the maximum learning rate. Defaults 0.01.
        clip_norm (bool, optional): whether to clip gradients by norm of 1. Default is True.
        scheduler (bool, optional): whether to use learning rate scheduler. Defaults to False.

    Returns:
        Tuple[nn.Module, Tuple[List[float], List[float], List[float], List[float]]]
    """
    training_acc, training_loss, testing_acc, testing_loss = list(), list(), list(), list()
    lr_hist = []

    # Define learning rate scheduler
    if scheduler:
        lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5,epochs],
                                         [0, lr_max, lr_max/20.0, 0])[0]

    # Define optimizer and criterion 
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    opt = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc, n = 0, 0, 0

        for i, (X,y) in enumerate(trainloader):
            model.train()
            X, y = X.cuda(), y.cuda()

            # Update learning rate
            if scheduler:
                lr = lr_schedule(epoch + (i + 1)/len(trainloader))
                opt.param_groups[0].update(lr=lr)
                lr_hist.append(lr)

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(X)
                loss = criterion(output, y)

            scaler.scale(loss).backward()

            if clip_norm:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
            
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        
        # Calculate testing accuracy and loss
        model.eval()
        test_loss, test_acc, m = 0, 0, 0

        with torch.no_grad():
            for X, y in testloader:
                X, y = X.cuda(), y.cuda()
                with torch.cuda.amp.autocast():
                    output = model(X)
                    test_loss += criterion(output, y).item() * y.size(0)
                    test_acc += (output.max(1)[1] == y).sum().item()

                    m += y.size(0)

        train_loss /= n
        train_acc /= n
        test_loss /= m
        test_acc /= m

        if scheduler:
            scheduler.step()

        print(f'VIT: Epoch: {epoch} | ',
              f'Train Acc: {train_acc:.4f}, ',
              f'Test Acc: {test_acc:.4f}, ',
              f'Time: {time.time() - start:.1f}, ',
              f'lr: {lr:.6f}')
    return model, training_acc, training_loss, testing_acc, testing_loss

def trainOneCLR(model, device, train_loader, criterion, scheduler, optimizer, use_l1=False, lambda_l1=0.01):
    """Function to train the model

    Args:
        model (instance): torch model instance of defined model
        device (str): "cpu" or "cuda" device to be used
        train_loader (instance): Torch Dataloader instance for trainingset
        criterion (instance): criterion to used for calculating the loss
        scheduler (function): scheduler to be used
        optimizer (function): optimizer to be used
        use_l1 (bool, optional): L1 Regularization method set True to use . Defaults to False.
        lambda_l1 (float, optional): Regularization parameter of L1. Defaults to 0.01.

    Returns:
        float: accuracy and loss values
    """
    model.train()
    pbar = tqdm(train_loader)
    lr_trend = []
    correct = 0
    processed = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch 
        # accumulates the gradients on subsequent backward passes. Because of this, when you start your training loop, 
        # ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)
        # Calculate loss
        loss = criterion(y_pred, target)

        l1=0
        if use_l1:
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
        loss = loss + lambda_l1*l1

        # Backpropagation
        loss.backward()
        optimizer.step()
        # updating LR
        if scheduler:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                lr_trend.append(scheduler.get_last_lr()[0])

        train_loss += loss.item()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        

        pbar.set_description(desc= f'Batch_id={batch_idx} Loss={train_loss/(batch_idx + 1):.5f} Accuracy={100*correct/processed:0.2f}%')
    return 100*correct/processed, train_loss/(batch_idx + 1), lr_trend


def testOneCLR(model, device, test_loader, criterion):
    """put model in eval mode and test it

    Args:
        model (instance): torch model instance of defined model
        device (str): "cpu" or "cuda" device to be used
        test_loader (instance): Torch Dataloader instance for testset
        criterion (instance): criterion to used for calculating the loss

    Returns:
        float: accuracy and loss values
    """
    model.eval()
    test_loss = 0
    correct = 0
    #iteration = len(test_loader.dataset)// test_loader.batch_size
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset), test_loss


def save_model(model, epoch, optimizer, path):
    """Save torch model in .pt format

    Args:
        model (instace): torch instance of model to be saved
        epoch (int): epoch num
        optimizer (instance): torch optimizer
        path (str): model saving path
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


def fit_model(net, optimizer, criterion, device, NUM_EPOCHS,train_loader, test_loader, use_l1=False, scheduler=None, save_best=False):
    """Fit the model

    Args:
        net (instance): torch model instance of defined model
        optimizer (function): optimizer to be used
        criterion (instance): criterion to used for calculating the loss
        device (str): "cpu" or "cuda" device to be used
        NUM_EPOCHS (int): number of epochs for model to be trained
        train_loader (instance): Torch Dataloader instance for trainingset
        test_loader (instance): Torch Dataloader instance for testset
        use_l1 (bool, optional): L1 Regularization method set True to use. Defaults to False.
        scheduler (function, optional): scheduler to be used. Defaults to None.
        save_best (bool, optional): If save best model to model.pt file, paramater validation loss will be monitered

    Returns:
        (model, list): trained model and training logs
    """
    training_acc, training_loss, testing_acc, testing_loss = list(), list(), list(), list()
    lr_trend = []
    if save_best:
        min_val_loss = np.inf
        save_path = 'model.pt'

    for epoch in range(1,NUM_EPOCHS+1):
        print("EPOCH: {} (LR: {})".format(epoch, optimizer.param_groups[0]['lr']))
        
        train_acc, train_loss, lr_hist = trainOneCLR(
            model=net, 
            device=device, 
            train_loader=train_loader, 
            criterion=criterion ,
            optimizer=optimizer, 
            use_l1=use_l1, 
            scheduler=scheduler
        )
        test_acc, test_loss = testOneCLR(net, device, test_loader, criterion)
        # update LR
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
        
        if save_best:
            if test_loss < min_val_loss:
                print(f'Valid loss reduced from {min_val_loss:.5f} to {test_loss:.6f}. checkpoint created at...{save_path}\n')
                save_model(net, epoch, optimizer, save_path)
                min_val_loss = test_loss
            else:
                print(f'Valid loss did not inprove from {min_val_loss:.5f}\n')
        else:
            print()

        training_acc.append(train_acc)
        training_loss.append(train_loss)
        testing_acc.append(test_acc)
        testing_loss.append(test_loss)
        lr_trend.extend(lr_hist)    

    if scheduler:   
        return net, (training_acc, training_loss, testing_acc, testing_loss, lr_trend)
    else:
        return net, (training_acc, training_loss, testing_acc, testing_loss)

class train:

    def __init__(self):

        self.train_losses = []
        self.train_acc    = []

    # Training
    def execute(self,net, device, trainloader, optimizer, criterion,epoch):

        #print('Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        #total = 0
        processed = 0
        pbar = tqdm(trainloader)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # get samples
            inputs, targets = inputs.to(device), targets.to(device)

            # Init
            optimizer.zero_grad()

            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            self.train_losses.append(loss.item())
            
            _, predicted = outputs.max(1)
            processed += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(desc= f'Epoch: {epoch},Loss={loss.item():3.2f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)


class test:

    def __init__(self):

        self.test_losses = []
        self.test_acc    = []

    def execute(self, net, device, testloader, criterion):

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(testloader.dataset)
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))

        # Save.
        self.test_acc.append(100. * correct / len(testloader.dataset))

def trainNetwork(net, device, trainloader, testloader, EPOCHS, lr=0.2):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    trainObj = train()
    testObj = test()

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        trainObj.execute(net, device, trainloader, optimizer, criterion, epoch)
        testObj.execute(net, device, testloader, criterion)
        scheduler.step()

    print('Finished Training')

    return trainObj, testObj