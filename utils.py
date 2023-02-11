import torch                   #PyTorch base libraries
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, ShiftScaleRotate, CoarseDropout, Cutout, PadIfNeeded
from albumentations.pytorch.transforms import ToTensorV2

class album_Compose_train():
    def __init__(self):
        self.albumentations_transform = Compose([
            PadIfNeeded(40),
            RandomCrop(32,32),
            HorizontalFlip(),
            Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], always_apply=True, p=1.00),
#            CoarseDropout(max_holes=3, max_height=8, max_width=8, min_holes=None, min_height=4, min_width=4, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], mask_fill_value=None, always_apply=False, p=0.7),
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class album_Compose_test():
    def __init__(self):
        self.albumentations_transform = Compose([
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])

    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class dataset_cifar10:
    """
    Class to load the data and define the data loader
    """

    def __init__(self, batch_size=128):

        # Defining CUDA
        cuda = torch.cuda.is_available()
        print("CUDA availability ?",cuda)

        # Defining data loaders with setting
        self.dataloaders_args = dict(shuffle=True, batch_size = batch_size, num_workers = 2, pin_memory = True) if cuda else dict(shuffle=True,batch_size = batch_size)
        self.sample_dataloaders_args = self.dataloaders_args.copy()

        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def data(self, train_flag):

        # Transformations data augmentation (only for training)
        if train_flag :
            return datasets.CIFAR10('./Data',
                            train=train_flag,
                            transform=album_Compose_train(),
                            download=True)

        # Testing transformation - normalization adder
        else:
            return datasets.CIFAR10('./Data',
                                train=train_flag,
                                transform=album_Compose_test(),
                                download=True)

    # Dataloader function
    def loader(self, train_flag=True):
        return(torch.utils.data.DataLoader(self.data(train_flag), **self.dataloaders_args))


    def data_summary_stats(self):
        # Load train data as numpy array
        train_data = self.data(train_flag=True).data
        test_data = self.data(train_flag=False).data

        total_data = np.concatenate((train_data, test_data), axis=0)
        print(total_data.shape)
        print(total_data.mean(axis=(0,1,2))/255)
        print(total_data.std(axis=(0,1,2))/255)

    def sample_pictures(self, train_flag=True, return_flag = False):

        # get some random training images
        images,labels = next(iter(self.loader(train_flag)))

        sample_size=25 if train_flag else 5

        images = images[0:sample_size]
        labels = labels[0:sample_size]

        fig = plt.figure(figsize=(10, 10))

        # Show images
        for idx in np.arange(len(labels.numpy())):
            ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
            npimg = unnormalize(images[idx])
            ax.imshow(npimg, cmap='gray')
            ax.set_title("Label={}".format(str(self.classes[labels[idx]])))

        fig.tight_layout()  
        plt.show()

        if return_flag:
            return images,labels

def unnormalize(img):
    channel_means = (0.4914, 0.4822, 0.4471)
    channel_stdevs = (0.2469, 0.2433, 0.2615)
    img = img.numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
         img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
    return np.transpose(img, (1,2,0))

#
def plot_loss_accuracy_graph(trainObj, testObj, EPOCHS):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    train_epoch_linspace = np.linspace(1, EPOCHS, len(trainObj.train_losses))
    test_epoch_linspace = np.linspace(1, EPOCHS, len(testObj.test_losses))

    # Loss Plot
    ax[0].plot(train_epoch_linspace, trainObj.train_losses, label='Training Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss vs. Epochs')
    ax[0].legend()

    ax2 = ax[0].twinx()
    ax2.plot(test_epoch_linspace, testObj.test_losses, label='Test Loss', color='red')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='center right')

    # Accuracy Plot
    ax[1].plot(train_epoch_linspace, trainObj.train_acc, label='Training Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy vs. Epochs')
    ax[1].legend()

    ax2 = ax[1].twinx()
    ax2.plot(test_epoch_linspace, testObj.test_acc, label='Test Accuracy', color='red')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='center right')

    plt.tight_layout()
    plt.show()


# define a function to plot misclassified images
def plot_misclassified_images(model, test_loader, classes, device):
    # set model to evaluation mode
    model.eval()

    misclassified_images = []
    actual_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    misclassified_images.append(data[i])
                    actual_labels.append(classes[target[i]])
                    predicted_labels.append(classes[pred[i]])

    # Plot the misclassified images
    fig = plt.figure(figsize=(12, 5))
    for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        npimg = unnormalize(misclassified_images[i].cpu())
        plt.imshow(npimg, cmap='gray', interpolation='none')
        sub.set_title("Actual: {}, Pred: {}".format(actual_labels[i], predicted_labels[i]),color='red')
    plt.tight_layout()
    plt.show()


def calAccuracy(net, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the  train images: {(100 * correct / total)} %%')

def calClassAccuracy(net, dataloader, classes, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


'''GradCAM in PyTorch.
Grad-CAM implementation in Pytorch
Reference:
[1] https://github.com/vickyliin/gradcam_plus_plus-pytorch
[2] The paper authors torch implementation: https://github.com/ramprs/grad-cam
'''

layer_finders = {}


def register_layer_finder(model_type):
    def register(func):
        layer_finders[model_type] = func
        return func
    return register


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


@register_layer_finder('resnet')
def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


# def denormalize(tensor, mean, std):
#     if not tensor.ndimension() == 4:
#         raise TypeError('tensor should be 4D')

#     mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
#     std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

#     return tensor.mul(std).add(mean)


# def normalize(tensor, mean, std):
#     if not tensor.ndimension() == 4:
#         raise TypeError('tensor should be 4D')

#     mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
#     std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

#     return tensor.sub(mean).div(std)


# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, tensor):
#         return self.do(tensor)

#     def do(self, tensor):
#         return normalize(tensor, self.mean, self.std)

#     def undo(self, tensor):
#         return denormalize(tensor, self.mean, self.std)

#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# class GradCAM:
#     """Calculate GradCAM salinecy map.
#     Args:
#         input: input image with shape of (1, 3, H, W)
#         class_idx (int): class index for calculating GradCAM.
#                 If not specified, the class index that makes the highest model prediction score will be used.
#     Return:
#         mask: saliency map of the same spatial dimension with input
#         logit: model output
#     A simple example:
#         # initialize a model, model_dict and gradcam
#         resnet = torchvision.models.resnet101(pretrained=True)
#         resnet.eval()
#         gradcam = GradCAM.from_config(model_type='resnet', arch=resnet, layer_name='layer4')
#         # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
#         img = load_img()
#         normed_img = normalizer(img)
#         # get a GradCAM saliency map on the class index 10.
#         mask, logit = gradcam(normed_img, class_idx=10)
#         # make heatmap from mask and synthesize saliency map using heatmap and img
#         heatmap, cam_result = visualize_cam(mask, img)
#     """

#     def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module):
#         self.model_arch = arch

#         self.gradients = dict()
#         self.activations = dict()

#         def backward_hook(module, grad_input, grad_output):
#             self.gradients['value'] = grad_output[0]

#         def forward_hook(module, input, output):
#             self.activations['value'] = output

#         target_layer.register_forward_hook(forward_hook)
#         target_layer.register_backward_hook(backward_hook)

#     @classmethod
#     def from_config(cls, arch: torch.nn.Module, model_type: str, layer_name: str):
#         target_layer = layer_finders[model_type](arch, layer_name)
#         return cls(arch, target_layer)

#     def saliency_map_size(self, *input_size):
#         device = next(self.model_arch.parameters()).device
#         self.model_arch(torch.zeros(1, 3, *input_size, device=device))
#         return self.activations['value'].shape[2:]

#     def forward(self, input, class_idx=None, retain_graph=False):
#         b, c, h, w = input.size()

#         logit = self.model_arch(input)
#         if class_idx is None:
#             score = logit[:, logit.max(1)[-1]].squeeze()
#         else:
#             score = logit[:, class_idx].squeeze()

#         self.model_arch.zero_grad()
#         score.backward(retain_graph=retain_graph)
#         gradients = self.gradients['value']
#         activations = self.activations['value']
#         b, k, u, v = gradients.size()

#         alpha = gradients.view(b, k, -1).mean(2)
#         # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
#         weights = alpha.view(b, k, 1, 1)

#         saliency_map = (weights*activations).sum(1, keepdim=True)
#         saliency_map = F.relu(saliency_map)
#         saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
#         saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
#         saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

#         return saliency_map, logit

#     def __call__(self, input, class_idx=None, retain_graph=False):
#         return self.forward(input, class_idx, retain_graph)

# def plotGradCAM(net, testloader, classes, device):

#     images, labels = next(iter(testloader))
#     gradcam = GradCAM.from_config(model_type='resnet', arch=net, layer_name='layer4')

#     fig = plt.figure(figsize=(5, 10))
#     idx_cnt=1
#     for idx in np.arange(len(labels.numpy())):

#         img = images[idx]
#         lbl = labels.numpy()[idx]

#         # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
#         img = img.unsqueeze(0).to(device)
#         org_img = denormalize(img,mean=(0.4914, 0.4822, 0.4471),std=(0.2469, 0.2433, 0.2615))

#         # get a GradCAM saliency map on the class index 10.
#         mask, logit = gradcam(img, class_idx=lbl)
#         # make heatmap from mask and synthesize saliency map using heatmap and img
#         heatmap, cam_result = visualize_cam(mask, org_img, alpha=0.4)

#         # Show images
#         # for idx in np.arange(len(labels.numpy())):
#         # Original picture
#         ax = fig.add_subplot(5, 3, idx_cnt, xticks=[], yticks=[])
#         npimg = np.transpose(org_img[0].cpu().numpy(),(1,2,0))
#         ax.imshow(npimg, cmap='gray')
#         ax.set_title("Label={}".format(str(classes[lbl])))
#         idx_cnt+=1

#         ax = fig.add_subplot(5, 3, idx_cnt, xticks=[], yticks=[])
#         npimg = np.transpose(heatmap,(1,2,0))
#         ax.imshow(npimg, cmap='gray')
#         ax.set_title("HeatMap".format(str(classes[lbl])))
#         idx_cnt+=1

#         ax = fig.add_subplot(5, 3, idx_cnt, xticks=[], yticks=[])
#         npimg = np.transpose(cam_result,(1,2,0))
#         ax.imshow(npimg, cmap='gray')
#         ax.set_title("GradCAM".format(str(classes[lbl])))
#         idx_cnt+=1

#     fig.tight_layout()  
#     plt.show()