import torch

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

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))

            running_loss += loss.item()
            num_iter += 1

    return running_loss / num_iter

def test(model=None, dataloader=None, device='cpu'):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to('cpu') == labels).sum().item()

    return float(correct / total)
