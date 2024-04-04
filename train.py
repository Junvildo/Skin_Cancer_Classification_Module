import torch
from tqdm import tqdm
from model import *
from data import *
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Arguments users used when running command lines
    parser.add_argument('--train-path', type=str, help='Where training data is located')
    parser.add_argument('--test-path', type=str, help='Where training data is located')
    parser.add_argument("--batch-size", default=32, type=int, help ="number of batch size")
    parser.add_argument('--epochs', default=10, type=int, help="number of epochs")
    parser.add_argument('--save-path', type=str, help="where to save the model")
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
    ])
    train_path = args.train_path
    test_path = args.test_path
    

    train_dataset = ImageDataset(data_dir = train_path,
                                    transform = transform)
    test_dataset = ImageDataset(data_dir = test_path,
                                    transform = transform)

    train_dataloader = DataLoader(dataset = train_dataset,
                                batch_size = args.batch_size,
                                shuffle = True)
    test_dataloader = DataLoader(dataset = test_dataset,
                                batch_size = args.batch_size,
                                shuffle = False)
    
    print(train_dataset.classes)
    model = EffNetB0()
    checkpoint = 'None'
    if args.ckpt != 'None':
        checkpoint = torch.load(args.ckpt)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9, eps=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)

    torch.manual_seed(69)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using {device}")
    print("-"*50)

    train_losses = []
    test_losses = []
    best_acc = -100
    start_epoch = 0

    if checkpoint != 'None':
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    for epoch in range(start_epoch, args.epochs):
        train_loss = 0.
        train_acc = 0.
        model.train()   
        for image, label in tqdm(train_dataloader, desc = "Training start: "):
            image = image.to(device)
            label = label.to(device)
            # Forward pass
            prediction = model(image)
            # Calculate  and accumulate loss
            loss = loss_fn(prediction, label)
            train_loss += loss.item()

            # Optimizer zero grad
            optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            pred_class = torch.argmax(torch.softmax(prediction, dim=1), dim=1)
            train_acc += (pred_class == label).sum().item()/len(prediction)

        scheduler.step()

        train_loss = train_loss / len(train_dataloader)
        train_acc /= len(train_dataloader)

        model.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.inference_mode():
            for image, label in tqdm(test_dataloader, desc = "Testing start: "):
                image = image.to(device)
                label = label.to(device)
                prediction = model(image)
                loss = loss_fn(prediction, label)
                test_loss += loss.item()
                pred_class = torch.argmax(torch.softmax(prediction, dim=1), dim=1)
                test_acc += (pred_class == label).sum().item()/len(prediction)
        test_loss = test_loss / len(test_dataloader)
        test_acc /= len(test_dataloader)

        print(f"Epochs {epoch+1}: Training loss: {train_loss:.2f} || Test loss: {test_loss:2f}")
        print(f"\t Train acc: {train_acc:2f} || Test acc: {test_acc:2f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.save_path, "last.pt"))
        if test_acc > best_acc:
            torch.save(checkpoint, os.path.join(args.save_path, "best.pt"))
            best_acc = test_acc


