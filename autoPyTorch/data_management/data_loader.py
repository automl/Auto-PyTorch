import os
import math
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms, utils



class DataLoader():
    def __init__(self):
        pass
        
    def load(self, url, size):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except:
            return None
        t = transforms.Compose([transforms.Resize(size),
                                transforms.CenterCrop(size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
        res = (t(img) * 255).int()
        return res.reshape((size*size*3))

    def save_imagenet_subset(self, root, name, class_wnids, image_size, max_images=None):
        with open(os.path.join(root, name) + '.data', 'w+') as data:
            with open(os.path.join(root, name) + '.label', 'w+') as label:
                for i, wnid in enumerate(class_wnids):
                    urls = requests.get('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + wnid).content
                    urls = urls.split(b"\n")
                    images = 0
                    for u in range(len(urls)):
                        if max_images is not None and images+1 > max_images / len(class_wnids):
                            break
                        img = self.load(urls[u].decode('utf-8'), image_size)
                        if img is None:
                            continue
                        images += 1
                        data.write(' '.join([str(rgb) for rgb in img.numpy()]) + '\n')
                        label.write(str(i) + '\n')
                    missing = math.floor(max_images/len(class_wnids)) - images 
                    if missing > 0:
                        print('Wnid', wnid, 'needs', missing, 'more images.')