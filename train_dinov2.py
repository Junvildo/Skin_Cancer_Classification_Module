import torch
import matplotlib.pyplot as plt
from torch import nn
from datasets import load_dataset
import albumentations as A
from torch.utils.data import DataLoader
from data_dinov2 import ClassificationDataset
from model_dinov2 import Dinov2ForImageClassification
import evaluate
from torch.optim import AdamW
from tqdm.auto import tqdm
import os
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Arguments users used when running command lines
    parser.add_argument('--train-path', type=str, help='Where training data is located')
    parser.add_argument('--val-path', default='', type=str, help='Where validation data is located')
    parser.add_argument('--test-path', type=str, help='Where test data is located')
    parser.add_argument("--batch-size", default=32, type=int, help ="number of batch size")
    parser.add_argument('--epochs', default=10, type=int, help="number of epochs")
    parser.add_argument('--num-class', default=2, type=int, help="number of classes")
    parser.add_argument('--save-path', type=str, help="where to save the model")
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    args = parser.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_dataset = load_dataset("imagefolder", data_dir=args.train_path)
    test_dataset = load_dataset("imagefolder", data_dir=args.test_path)
    if args.val_path != '':
        validation_dataset = load_dataset("imagefolder", data_dir=args.val_path)
    else:
        validation_dataset = test_dataset

    train_transform = A.Compose([
    A.Resize(width=224, height=224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ClassificationDataset(train_dataset["train"], transform=train_transform)
    val_dataset = ClassificationDataset(validation_dataset["train"], transform=test_transform)
    test_dataset = ClassificationDataset(test_dataset["train"], transform=test_transform)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    def collate_fn(inputs):
        batch = dict()
        batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
        batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
        batch["original_images"] = [i[2] for i in inputs]

        return batch

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=args.num_class)

    for name, param in model.named_parameters():
        if name.startswith("dinov2"):
            param.requires_grad = False

    
    metric = evaluate.load("accuracy")


    learning_rate = 5e-5
    epochs = args.epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    model.train()

    # Initialize variables to track the best accuracy and corresponding model state
    best_val_accuracy = 0.0
    best_model_path = os.path.join(args.save_path, "best_model.pt")
    last_model_path = os.path.join(args.save_path, "last_model.pt")

    for epoch in range(epochs):
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # forward pass
            outputs = model(pixel_values, labels=labels)
            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()
            


            # evaluate
            with torch.no_grad():
                predicted = outputs.logits.argmax(dim=1)

            # note that the metric expects predictions + labels as numpy arrays
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

            # let's print loss and metrics every 100 batches
            if idx % 50 == 0:
                metrics = metric.compute()

                print("Loss:", loss.item())
                print("Accuracy:", metrics["accuracy"])
                



        # Validation step
        model.eval()
        val_metric = evaluate.load("accuracy")  # Initialize the same metric you used for training
        for val_batch in tqdm(val_dataloader):
            val_pixel_values = val_batch["pixel_values"].to(device)
            val_labels = val_batch["labels"].to(device)

            with torch.no_grad():
                val_outputs = model(val_pixel_values)
                val_logits = val_outputs.logits
                val_predicted = val_logits.argmax(dim=1)

            val_metric.add_batch(predictions=val_predicted.detach().cpu().numpy(), references=val_labels.detach().cpu().numpy())

        val_metrics = val_metric.compute()
        print("Validation Accuracy:", val_metrics["accuracy"])

        # Save the best model based on validation accuracy
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with validation accuracy: {best_val_accuracy}")

        model.train()

    # Save the last model at the end of the last epoch
    torch.save(model.state_dict(), last_model_path)
    print(f"Saved last model at the end of training")

    del model

    def evaluate_model(model, dataloader, device):
        # put model in evaluation mode
        model.eval()

        # initialize metric for testing
        test_metric = evaluate.load("accuracy")  # Initialize the same metric you used for training
        all_predictions = []
        all_labels = []

        for idx, batch in enumerate(tqdm(dataloader)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # forward pass
            with torch.no_grad():
                outputs = model(pixel_values)
                predicted = outputs.logits.argmax(dim=1)
                all_predictions.extend(predicted.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

            # note that the metric expects predictions + labels as numpy arrays
            test_metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # compute and print the final test metrics
        test_metrics = test_metric.compute()
        accuracy = test_metrics["accuracy"]
        auc_score = roc_auc_score(all_labels, all_predictions)
        fpr, tpr, _ = roc_curve(all_labels, all_predictions)
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        return accuracy, auc_score, fpr, tpr, precision, recall, f1_score

    # Load the best model
    best_model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=args.num_class)  # Replace with your model class
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)

    # Load the last model
    last_model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base", hidden_size=768, num_labels=args.num_class)  # Replace with your model class
    last_model.load_state_dict(torch.load(last_model_path))
    last_model.to(device)

    # Evaluate the best model
    best_accuracy, best_auc, best_fpr, best_tpr, best_precision, best_recall, best_f1 = evaluate_model(best_model, test_dataloader, device)
    print("Best Model Test Accuracy:", best_accuracy)
    print("Best Model AUC Score:", best_auc)
    print("Best Model Precision:", best_precision)
    print("Best Model Recall:", best_recall)
    print("Best Model F1 Score:", best_f1)
    del best_model
    # Evaluate the last model
    last_accuracy, last_auc, last_fpr, last_tpr, last_precision, last_recall, last_f1 = evaluate_model(last_model, test_dataloader, device)
    print("Last Model Test Accuracy:", last_accuracy)
    print("Last Model AUC Score:", last_auc)
    print("Last Model Precision:", last_precision)
    print("Last Model Recall:", last_recall)
    print("Last Model F1 Score:", last_f1)
    del last_model
    # Plotting AUC Curves
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(best_fpr, best_tpr, color='blue', lw=2, label='Best Model (AUC = %0.2f)' % best_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Best Model ROC Curve')
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(last_fpr, last_tpr, color='green', lw=2, label='Last Model (AUC = %0.2f)' % last_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Last Model ROC Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()