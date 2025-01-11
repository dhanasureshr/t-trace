import torch.nn as nn
import torch.nn.functional as F

# Define a simple classifier
class sentiment_classifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(sentiment_classifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)  # Fully connected layer

    def forward(self, x):
        return self.fc(x)
