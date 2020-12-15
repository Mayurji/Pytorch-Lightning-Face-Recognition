import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from model import SiameseNetwork
%matplotlib inline

model = SiameseNetwork.load_from_checkpoint("siamese_face_recognition.ckpt")

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


# def view_dissimilarity(testDirectory):
#     folder_dataset_test = dset.ImageFolder(root=testDirectory)
#     siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
#                                             transform=transforms.Compose([transforms.Resize((100,100)),
#                                                                           transforms.ToTensor()
#                                                                           ])
#                                            ,should_invert=False)

#     test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
#     dataiter = iter(test_dataloader)
#     x0,_,_ = next(dataiter)

#     for i in range(4):
#         _,x1,label2 = next(dataiter)
#         concatenated = torch.cat((x0,x1),0)
#         model.eval()
#         output1,output2 = model(Variable(x0).cuda(),Variable(x1).cuda())
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))