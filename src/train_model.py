import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

CHAR_SET = "2346789ACDEFGHJKLNPQRTUVXYZ"
NUM_CLASSES = len(CHAR_SET)
SEQ_LENGTH = 5
IMAGE_WIDTH, IMAGE_HEIGHT = 200, 60

char_to_idx = {char: idx for idx, char in enumerate(CHAR_SET)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}


class CaptchaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        self.transform = (
            transform
            if transform
            else transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        label_str = os.path.splitext(filename)[0].upper()

        image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        image = self.transform(image)

        label = torch.tensor([char_to_idx[c] for c in label_str], dtype=torch.long)
        return image, label


class CaptchaCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(256, 256, kernel_size=(7, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.rnn_input_size = 256
        self.rnn_hidden_size = 512
        self.lstm = nn.LSTM(
            self.rnn_input_size,
            self.rnn_hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.5,
            batch_first=True,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(SEQ_LENGTH)
        self.classifier = nn.Linear(self.rnn_hidden_size * 2, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        output, _ = self.lstm(x)

        output = output.permute(0, 2, 1)
        output = self.adaptive_pool(output)
        output = output.permute(0, 2, 1)

        output = self.classifier(output.contiguous().view(-1, self.rnn_hidden_size * 2))
        output = output.view(-1, SEQ_LENGTH, NUM_CLASSES)

        return output


def train_model():
    data_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )

    full_dataset = CaptchaDataset("dataset/clean", transform=data_transform)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    BATCH_SIZE = 12
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )

    model = CaptchaCRNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    EPOCHS = 80
    best_val_accuracy = 0.0

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = 0
            for i in range(SEQ_LENGTH):
                loss += criterion(outputs[:, i, :], labels[:, i])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(2)
            correct_train += (predicted == labels).all(dim=1).sum().item()
            total_train += labels.size(0)
            train_bar.set_postfix(
                loss=total_loss / (train_bar.n + 1), acc=correct_train / total_train
            )

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = 0
                for i in range(SEQ_LENGTH):
                    loss += criterion(outputs[:, i, :], labels[:, i])
                total_val_loss += loss.item()

                _, predicted = outputs.max(2)
                correct_val += (predicted == labels).all(dim=1).sum().item()
                total_val += labels.size(0)
                val_bar.set_postfix(
                    loss=total_val_loss / (val_bar.n + 1), acc=correct_val / total_val
                )

        avg_val_loss = total_val_loss / len(val_dataloader)
        epoch_val_accuracy = correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_accuracy:.2%} | Val Loss: {avg_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.2%}"
        )

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), "captcha_crnn_best.pth")
            print(f"⭐ Saved best model with Val Accuracy: {best_val_accuracy:.2%}")

    print("\n--- Loading best model for final evaluation ---")
    model.load_state_dict(torch.load("captcha_crnn_best.pth"))
    print("\n--- Evaluating on Test Set ---")
    evaluate_model(model, test_dataloader, device)

    print("✅ Model training complete.")

    # --- Plotting section ---
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()
    plt.close()

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    eval_bar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for images, labels in eval_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(2)
            correct += (predicted == labels).all(dim=1).sum().item()
            total += labels.size(0)
            eval_bar.set_postfix(accuracy=correct / total)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")
    return accuracy


def predict_captcha(image_path, model, device):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(2)

    predicted_chars = [idx_to_char[idx.item()] for idx in predicted.squeeze(0)]
    return "".join(predicted_chars)


if __name__ == "__main__":
    train_model()
    print("\n--- Testing Prediction Function ---")
    model = CaptchaCRNN()
    if os.path.exists("captcha_crnn_best.pth"):
        model.load_state_dict(torch.load("captcha_crnn_best.pth"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        full_dataset = CaptchaDataset("dataset/clean")
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        _, _, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        if len(test_dataset) > 0:
            original_idx = test_dataset.indices[0]
            sample_image_filename = full_dataset.image_files[original_idx]
            sample_image_path = os.path.join("dataset/clean", sample_image_filename)
            print(f"Predicting for sample image from test set: {sample_image_path}")
            predicted_captcha = predict_captcha(sample_image_path, model, device)
            print(f"Predicted Captcha: {predicted_captcha}")
            print(f"Actual Label: {os.path.splitext(sample_image_filename)[0].upper()}")
        else:
            print("Test set is empty, cannot perform prediction test.")
    else:
        print("No 'captcha_crnn_best.pth' found. Train the model first.")
