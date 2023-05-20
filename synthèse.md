# Liste des choses à faire pour une étude de deeplearning

## <h2 style='color:red'>Construction manuelle </h2>

### 1. Télécharger les données et  définir les chemins d'accès.

```python
from pathlib import Path
import zipfile
import requests
import os

data_path= Path("data")
images_path = data_path /"pizza_steak_sushi"

images_path.mkdir(parents=True, exist_ok=True)


with open(data_path/"pizza_steak_sushi.zip","wb") as f:
    requete= requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    f.write(requete.content)

with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip","r") as f:
    f.extractall(images_path)

os.remove(data_path/"pizza_steak_sushi.zip")
```

### 2. Passez les données sous forme de dataloader.(jetaime enormement)

- Définir un transformeur 
- Passer les données dans un image folder pour former les datasets
- Passer ls datasets avec les caractéristiques voulu

```python

Transformer = transforms.Compose([
  transforms.Resize((64, 64)),                                     
  transforms.ToTensor()])

def make_dataloader(train_path,test_path,transformer,batch_size):

    trainset = datasets.ImageFolder(root=train_path,
                                    transform=transformer)
    
    testset = datasets.ImageFolder(root=test_path,
                                    transform=transformer)

    train_dl=DataLoader(dataset=trainset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=os.cpu_count())
    
    test_dl=DataLoader(dataset=testset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=os.cpu_count())
    
    return train_dl, test_dl 
```

### 3. Définir son modèle

- Donner l'architecture souhaitée.
- Mettre les input, hidden, output neurones en paramètres.
- Ne pas oublier de définir la fonction forward.

```python
import torch
from torch import nn 

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.block_2(self.block_1(x)))
```

# 4. Entraîner et tester le modèle sur les données

- Définir une loss_fn et un optimizer
- Définir le train step
- Définir le test step
- Regrouper les deux pour sortir les stats

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)

def train_step(model:nn.Module,
               loss_fn:nn.Module,
               optimizer:torch.optim,
               device,
               dataloader:torch.utils.data.DataLoader):

    model.train()
    
    train_loss = 0
    train_acc = 0


    for batch, (x,y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss=loss_fn(pred,y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        pred_label = torch.argmax(torch.softmax(pred,dim=1),dim=1)
        train_acc += (pred_label == y).sum().item()/len(pred)

    train_loss = train_loss /len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(model:nn.Module,
              loss_fn,device,
              dataloader):

    model.eval()
    test_loss=0
    test_acc=0

    with torch.inference_mode():
        for batch, (x,y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = loss_fn(pred,y)
            test_loss += loss.item()
            
            label = torch.argmax(torch.softmax(pred,dim=1),dim=1)
            test_acc += (label == y).sum().item()/len(pred)

    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)

    return test_loss, test_acc


def train(model,train_dataloader,test_dataloader,device,optimizer,loss_fn,epochs):

    from tqdm.auto import tqdm
    for epoch in tqdm(range(epochs)):

        results={"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

        train_loss,train_acc=train_step(model=model,
                                        dataloader=train_dataloader,
                                        device=device,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer)
        
        test_loss, test_acc=test_step(model=model,
                                      loss_fn=loss_fn,
                                      device=device,
                                      dataloader=test_dataloader,
                                      )
        

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


    return results
```

### 5. Sauvegarder le modèle

- Définir un nom de modèle
- Définir une localisation pour le stockage
- Sauvegarder selon les paramètres

```python
def save_model(model:nn.Module,nom_modele):

    models_path=Path("D:\Adam\Pytorch\exercises")
    nom_du_modele = nom_modele.pt

    model_path=models_path/nom_du_modele

    torch.save(obj=model.state_dict(),f=model_path)
```

### 6. Charger le modèle

- Définir une localisation

```python
def load_model(filepath=args.model_path): 
  model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=128,
                                output_shape=3).to(device)

  print(f"[INFO] Loading in model from: {filepath}")
                           
  model.load_state_dict(torch.load(filepath))

  return model
```

## <h2 style='color:red'>Imporation d'un modèle préentrainé </h2>

### 1. Définir le bon transformer

- L'importer automatiquement
- Le construire selon les indications suivies

```python

import torchvision
from torchinfo import summary

#Automatiquement
weights= torchvision.models.ResNet101_Weights.DEFAULT
transformer = weights.transforms()

#Manuellement
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
```

### 2. Importer le modèle

- Load le modèle de torchvision ou aure localisation
- Ajuster le modèle
- Visualiser pour mieux modifier

```python
model = torchvision.models.resnet101(pretrained=True).to(device)


for param in model.parameters():
    param.requires_grad=False


model.fc=nn.Sequential(
    nn.Linear(in_features=2048, out_features=3, bias=True))

summary(model=model,
        input_size=(1,3,224,224),
        col_names=["input_size",'output_size','num_params','trainable'],
        row_settings=['var_names'])
```

### 3. Train le modèle sur nos données

- Utiliser nos données pour rendre le modèle performant pour le projet.
- Evaluer les performances

```python
model_trained= train(model=model,
                     train_dataloader=train_dataloader,
                     test_dataloader=test_dataloader,
                     device=device,
                     optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001),
                     loss_fn=loss_fn,
                     epochs=5)
```
### 4. Tester sur plusieurs photos

```python
liste_images_paths=list(image_path.glob("*/*/*.jpg"))

    for path in liste_images_paths:

    image = Image.open(path)
    image_trans = transformer(image).unsqueeze(0)

    pred = model(image_trans.to(device))
    prob= torch.softmax(pred,dim=1).max().item()

    label = torch.argmax(torch.softmax(pred,dim=1),dim=1).item()
    vrai_label = class_names[label]

    vrai_label, prob
```

## <h2 style='color:red;'>Bonus</h2>

### 1. Faire une confusion matrix
- Aide à identifier les classes qui posent problème visuellement

```python
########################################################
#Définir la liste des prédictions
model_0.eval()

liste_pred=[]
with torch.inference_mode():
    for (x,y) in test_dataloader:
        x = x.to(device)
        y = y.to(device)

        pred=model_0(x)
        label=torch.argmax(torch.softmax(pred,dim=1),dim=1)
        liste_pred.append(label)

liste_pred = torch.cat(liste_pred)

########################################################
#Définir la liste des labels réels

liste_true = torch.cat([y for x,y in test_dataloader])

########################################################
#Plot la confusion matrix

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

conf_max=ConfusionMatrix(task='multiclass',num_classes=len(class_names))

conf_max_tens=conf_max(preds=liste_pred.to("cpu"),
                       target=liste_true.to("cpu"))


plot_confusion_matrix(conf_max_tens.numpy(),
                      class_names=class_names)
```

### 2. Mettre les résultats en dataframe

```python

########################################################
#Définir une liste de dictionnaires puis la passer en dataframe

from pathlib import Path
from PIL import Image
import pandas as pd


liste_images_paths=list(image_path.glob("*/*/*.jpg"))
liste_images_paths
liste_totale=[]
for path in liste_images_paths:
    dico_path={}
    vrai_nom=path.parent.stem

    dico_path["path"] = path
    dico_path["label"] = vrai_nom

    image = Image.open(path)
    image=simple_transform(image).unsqueeze(0)

    model_0.eval()
    with torch.inference_mode():
        pred = model_0(image.to(device))
        prob = torch.softmax(pred,dim=1)
        label_pred= class_names[torch.argmax(prob,dim=1)]

        dico_path['prob']=prob.max().item()
        dico_path['predicition']=label_pred
        dico_path["correct"]= vrai_nom == label_pred

    liste_totale.append(dico_path)


df = pd.DataFrame(liste_totale)
```
### 3. Lier son modèle à Mlflow

- Changer la fonction train
- Créer un write et train

```python
#Changement de fonction
def train(model,train_dataloader,test_dataloader,device,optimizer,loss_fn,epochs,writer):
    import mlflow
    from tqdm.auto import tqdm
    mlflow.set_tracking_uri("http://localhost:5000")
    for epoch in tqdm(range(epochs)):

        results={"train_loss": 0,
               "train_acc": 0,
               "test_loss": 0,
               "test_acc": 0}

        train_loss,train_acc=train_step(model=model,
                                        dataloader=train_dataloader,
                                        device=device,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer)
        
        test_loss, test_acc=test_step(model=model,
                                      loss_fn=loss_fn,
                                      device=device,
                                      dataloader=test_dataloader,
                                      )
        

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"] = train_loss
        results["train_acc"] = train_acc
        results["test_loss"] = test_loss
        results["test_acc"] = test_acc

        mlflow.log_metrics(results,step=epoch)

        if writer:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_acc", train_acc, epoch)
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_acc", test_acc, epoch)

    if writer:
        # Get the log directory from the writer
        log_dir = writer.log_dir

        # Log the entire model
        mlflow.pytorch.log_model(model, "model")

        # Log the writer's log directory as an artifact
        mlflow.log_artifact(log_dir, artifact_path="tensorboard_logs")

    return results

#Créer un writer et train

date = datetime.now().strftime("%Y-%m-%d")

log_dir = os.path.join("runs",date,"test")
summary=SummaryWriter(log_dir=log_dir)

training_model= train(model=model,
                      train_dataloader=train_dl,
                      test_dataloader=test_dl,
                      device=device,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=5,
                      writer=summary)
