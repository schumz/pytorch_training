import torch
from torch import nn


def train_step(model:nn.Module,loss_fn:nn.Module,optimizer:torch.optim,device,dataloader:torch.utils.data.DataLoader):

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


def test_step(model:nn.Module,loss_fn,device,dataloader):

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
