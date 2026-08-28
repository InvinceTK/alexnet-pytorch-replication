
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
from datasets import load_dataset
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
from torchvision.transforms import v2
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 128
DEVICE= 'cuda' if torch.cuda.is_available() else 'mps'
IMAGE_DIM = 227 
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 100  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use

class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity = 'relu')
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity = 'relu')
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        # nn.init.constant_(self.net[4].bias, 1)
        # nn.init.constant_(self.net[10].bias, 1)
        # nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)

mini_imagenet_classes = [
    "house finch, linnet, Carpodacus mexicanus",
    "robin, American robin, Turdus migratorius",
    "triceratops",
    "green mamba",
    "harvestman, daddy longlegs, Phalangium opilio",
    "toucan",
    "goose",
    "jellyfish",
    "nematode, nematode worm, roundworm",
    "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
    "dugong, Dugong dugon",
    "Walker hound, Walker foxhound",
    "Ibizan hound, Ibizan Podenco",
    "Saluki, gazelle hound",
    "golden retriever",
    "Gordon setter",
    "komondor",
    "boxer",
    "Tibetan mastiff",
    "French bulldog",
    "malamute, malemute, Alaskan malamute",
    "dalmatian, coach dog, carriage dog",
    "Newfoundland, Newfoundland dog",
    "miniature poodle",
    "white wolf, Arctic wolf, Canis lupus tundrarum",
    "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
    "Arctic fox, white fox, Alopex lagopus",
    "lion, king of beasts, Panthera leo",
    "meerkat, mierkat",
    "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
    "rhinoceros beetle",
    "ant, emmet, pismire",
    "black-footed ferret, ferret, Mustela nigripes",
    "three-toed sloth, ai, Bradypus tridactylus",
    "rock beauty, Holocanthus tricolor",
    "aircraft carrier, carrier, flattop, attack aircraft carrier",
    "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
    "barrel, cask",
    "beer bottle",
    "bookshop, bookstore, bookstall",
    "cannon",
    "carousel, carrousel, merry-go-round, roundabout, whirligig",
    "carton",
    "catamaran",
    "chime, bell, gong",
    "clog, geta, patten, sabot",
    "cocktail shaker",
    "combination lock",
    "crate",
    "cuirass",
    "dishrag, dishcloth",
    "dome",
    "electric guitar",
    "file, file cabinet, filing cabinet",
    "fire screen, fireguard",
    "frying pan, frypan, skillet",
    "garbage truck, dustcart",
    "hair slide",
    "holster",
    "horizontal bar, high bar",
    "hourglass",
    "iPod",
    "lipstick, lip rouge",
    "miniskirt, mini",
    "missile",
    "mixing bowl",
    "oboe, hautboy, hautbois",
    "organ, pipe organ",
    "parallel bars, bars",
    "pencil box, pencil case",
    "photocopier",
    "poncho",
    "prayer rug, prayer mat",
    "reel",
    "school bus",
    "scoreboard",
    "slot, one-armed bandit",
    "snorkel",
    "solar dish, solar collector, solar furnace",
    "spider web, spider's web",
    "stage",
    "tank, army tank, armored combat vehicle, armoured combat vehicle",
    "theater curtain, theatre curtain",
    "tile roof",
    "tobacco shop, tobacconist shop, tobacconist",
    "unicycle, monocycle",
    "upright, upright piano",
    "vase",
    "wok",
    "worm fence, snake fence, snake-rail fence, Virginia fence",
    "yawl",
    "street sign",
    "consomme",
    "trifle",
    "hotdog, hot dog, red hot",
    "orange",
    "cliff, drop, drop-off",
    "coral reef",
    "hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa",
    "ear, spike, capitulum",
]

transforms = v2.Compose([
          v2.RGB(),
          v2.CenterCrop(IMAGE_DIM),
          v2.ToTensor(),
          v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
class ImageNet(Dataset):

    def __init__(self):
        self.labels = ds["label"]
        self.imgs = ds["image"]

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                img = transforms(self.imgs[idx])
                label = torch.tensor(self.labels[idx])
                return img, label
            except Exception as e:
                print(f"Failed idx={idx}: {e}", flush=True)
                idx = (idx + 1) % len(self)
        raise RuntimeError(f"10 consecutive samples failed near idx={idx}")
    
@torch.no_grad()  
def evaluate_model(dataloader):
    model.to(DEVICE)
    total_steps = 0 
    accuracies = []
    for img, label in dataloader:
        total_steps+=1 
        img, label = img.to(DEVICE), label.to(DEVICE)
        out = model(img)
        
        _, preds = torch.max(out, dim=1)
        
        # preds = 32 * [1], [2], [3]
        # label = 32 * [1], [3], [4]
        # preds == label = 32 * [True], [False]
        # print(preds.shape), print(label.shape)
        buffer = (preds == label)
        acc = (torch.sum(buffer) / len(buffer)).cpu().item()
        accuracies.append(acc)
        
        if total_steps % 10 == 0 :
            print("step ={} , acc = {}".format(total_steps, acc))
        
    return accuracies

from PIL import Image
import matplotlib.pyplot as plt

def display_img(img, out="out.png"):
    original_img = ds[0]['image']  # PIL image, (H, W, C)-ish

    # Transformed tensor (C, H, W) → (H, W, C) for matplotlib
    transformed_img = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(transformed_img)
    axes[1].set_title("Transformed")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    
@torch.inference_mode()
def perform_inference(img):
    display_img(img)

    img = img.to(DEVICE)

    output = model(img)                          # (1, 100) logits
    probs = F.softmax(output, dim=1)               # (1, 100) probabilities

    top5_probs, top5_idx = probs[0].topk(5)        # (5,), (5,)

    print(top5_idx.tolist())
    print(top5_probs.tolist())
    # for prob, idx in zip(top5_probs.tolist(), top5_idx.tolist()):
     
    #     name = mini_imagenet_classes[idx]
    #     print(f"{name:60s} {prob * 100:.2f}%")
    
    
import argparse
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="choose whether to evaluate or inference")
    args = parser.parse_args()
    
    # pprint({k : type(v) for k,v in model.items()})
    model = AlexNet()
    model.to(DEVICE)
    ds = load_dataset('timm/mini-imagenet', split = 'test')
    dictionary = torch.load('alexnet_data_out/models/alexnet_states_e47.pkl')

    if args.mode == 'e':
        load_dotenv()
        HF_TOKEN = os.getenv('HF_TOKEN')
        
        print(len(ds))
        dataset = ImageNet()
        model.load_state_dict(dictionary['model'])
        model.eval()
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE)
        accuracies = evaluate_model(dataloader)
        print(accuracies[::10])
    
    if args.mode =='i':
        print('performing inference')
        dataset = ImageNet()
        dataloader = DataLoader(dataset, batch_size = 1)
        img,label = next(iter(dataloader))
        perform_inference(img)
        
       
        
        
    
