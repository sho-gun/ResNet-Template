import torch.optim as optim

def train(model=None, dataloader=None, criterion=None, optimizer=None, device='cpu'):
    running_loss = 0.0
    num_iter = 0
    max_iter = len(dataloader)

    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_iter += 1

        if (i+1) % 100 == 0:
            print('Iter [{}/{}], Loss = {}'.format(i + 1, max_iter, running_loss / num_iter))

    return running_loss / num_iter

def val(model=None, dataloader=None, criterion=None, device='cpu'):
    running_loss = 0.0
    num_iter = 0

    for inputs, labels in dataloader:
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))

        running_loss += loss.item()
        num_iter += 1

    return running_loss / num_iter
