import torch
#import protonets.utils.data as data_utils
#import protonets.utils.model as model_utils
import torchvision
import tqdm
import random
import numpy 
import conductance 
from quantize import quantize

def binlist2int(exp,mylist):
    return [ exp**x for x in mylist ]

def euclidean(x, y, qbits):
    xnorm = torch.norm(x)
    ynorm = torch.norm(y)
    x = x - x.mean()
    y = y - y.mean()
    x = quantize(x/xnorm,qbits)
    y = quantize(y/ynorm,qbits)
    z = x - y #quantize(x-y,qbits).abs()
    z = torch.pow(z, 2) #quantize(torch.pow(z, 2),qbits)
    return z.sum().item() 

def manhattan(x, y, qbits):
    xnorm = torch.norm(x)
    ynorm = torch.norm(y)
    x = quantize(x/xnorm,qbits)
    y = quantize(y/ynorm,qbits)
    z = x-y #quantize(x-y,qbits).abs()    
    return z.abs().sum().item() 

def chebyshev(x, y, qbits):
    xnorm = torch.norm(x)
    ynorm = torch.norm(y)
    x = quantize(x/xnorm,qbits)
    y = quantize(y/ynorm,qbits)
    z = x-y #quantize(x-y,qbits)    
    return z.abs().max().item() 

def cosine(x, y, qbits):
    xnorm = torch.norm(x)
    ynorm = torch.norm(y)
    x = quantize(x/xnorm,qbits)
    y = quantize(y/ynorm,qbits)
    z = x*y #quantize(x*y,qbits)
    z = 1-z.sum()
    return z.item() 

def dot(x, y, qbits):
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    z = quantize(x-y,qbits)    
    z = x*y #quantize(x*y,qbits)
    z = 1-z.sum()
    return z.item() 

def mcam(x, y, qbits):
    if qbits == 3:
        Gb = conductance.G_3bit
    elif qbits == 4:
        Gb = conductance.G_4bit
    else:
        raise Exception("MCAM only supports quantization bits up to 4") 
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    x = x.data.numpy()
    y = y.data.numpy()
    G = numpy.zeros(len(x))
    for i in range(len(x)):        
        G[i] = Gb[int(y[i])][int(x[i])]
    d = numpy.sum(G)

    return d 

def mcam_ideal(x, y, qbits):
    if qbits == 3:
        Gb = conductance.G_3bit
    elif qbits == 4:
        Gb = conductance.G_4bit
    else:
        raise Exception("MCAM only supports quantization bits up to 4") 
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    x = x.data.numpy()
    y = y.data.numpy()
    G = numpy.zeros(len(x))
    for i in range(len(x)):        
        G[i] = Gb[0][numpy.abs(int(y[i])-int(x[i]))]
    d = numpy.sum(G)

    return d 


#torch.pow(x-y,1).abs().sum().item() 

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d

def get_values(values, k, s):
    v = [i for i,x in enumerate(values) if x == k]
    return random.sample(v,s)

def main(opt):
    
    dataset = torchvision.datasets.Omniglot(
        root="./data", download=True, background=False,
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([28,28]),
                torchvision.transforms.ToTensor(),
            ])    
    )
    #print(len(dataset))

    image, label = dataset[0]
    #print(image.size())  # torch.Tensor
    #print(type(label))  # int


    model = torch.load(opt['model.model_path'])
    model.eval()
 
    #print(model.encoder)

    x = model.encoder.forward(image[-1,:,:].reshape([1,1,28,28]))
    #print(x)
    
    nlen = [int(0.8*len(dataset)),int(0.2*len(dataset))]
    #print(nlen)
    trainset, testset = torch.utils.data.random_split(dataset, nlen)

    dataset = 0

    train_values = []
    for d in tqdm.tqdm(trainset):
        train_values.append(d[1])
    test_values = []
    for d in tqdm.tqdm(testset):
        test_values.append(d[1])
    
    n_way = opt['data.test_way'] 
    n_shot = opt['data.test_shot']  

    acc = 0
    dist = opt['dist.distance']
    for it in tqdm.tqdm(range(opt['data.test_episodes'])):

        k = random.sample(train_values, n_way)
        q = random.sample(k, 1)
        while not (q[0] in test_values):
            q = random.sample(k, 1)

        support = []
        support_val = []
        for i in k:
            s = get_values(train_values, i, n_shot)
            for j in s:
                x = model.encoder.forward(1-trainset[j][0][-1,:,:].reshape([1,1,28,28]))
                support.append(x)
                support_val.append(i)
                
        s = get_values(test_values, q[0], 1)
        x = model.encoder.forward(1-testset[s[0]][0][-1,:,:].reshape([1,1,28,28])) 
        query = x        

        d = [0]*(n_way*n_shot)
        for i in range(n_way*n_shot):
            if dist == "euclidean":
                d[i] = euclidean(support[i][0],query[0],opt['dist.qbits']) 
            elif dist == "manhattan":
                d[i] = manhattan(support[i][0],query[0],opt['dist.qbits']) 
            elif dist == "cosine":
                d[i] = cosine(support[i][0],query[0],opt['dist.qbits']) 
            elif dist == "dot":
                d[i] = dot(support[i][0],query[0],opt['dist.qbits']) 
            elif dist == "chebyshev":
                d[i] = chebyshev(support[i][0],query[0],opt['dist.qbits']) 
            elif dist == "mcam":
                d[i] = mcam(support[i][0],query[0],opt['dist.qbits']) 
            else:
                d[i] = mcam_ideal(support[i][0],query[0],opt['dist.qbits']) 
        if q[0] == support_val[int(numpy.floor(numpy.argmin(d)))]:
            acc = acc + 1
    
    print("------------------")
    print("N-way   : ", opt['data.test_way'])
    print("K-shot  : ", opt['data.test_shot'])
    print("Episode : ", opt['data.test_episodes'])
    print("Distance: ", opt['dist.distance'])
    print("Q bits  : ", opt['dist.qbits'])
    print("Accuracy:", acc*100/opt['data.test_episodes'], '%')





