import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

# 加载基因表达数据


#import data from csv

#if using gene expression data or not
gene_expr = True
# gene_expr = False

#if using feature selected data or not
# full = True
full = False


if gene_expr:
    if full:
        gene_data = pd.read_csv(r'./G5_lung_gene-expr.csv')
        feat = 20531
        data = gene_data
    else: 
        gene_data_subset = pd.read_csv(r'./G5_lung_gene-expr_selected_features.csv')
        feat = 350
        data = gene_data_subset
else: 
    if full:
        dna_meth_data = pd.read_csv(r'./G5_lung_dna-meth.csv')
        feat = 5000
        data = dna_meth_data
    else:
        dna_meth_data_subset = pd.read_csv(r'./G5_lung_dna-meth_selected_features.csv')
        feat = 294
        data = dna_meth_data_subset
        
        






start = time.time() #for timing the code
X = data.iloc[:, 2:]
y = data.iloc[:, 1]
y = pd.DataFrame(y)
y = y.replace({'Primary Tumor': 1, 'Solid Tissue Normal': 2})

# split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.preprocessing import StandardScaler
#log scaling only on gene expression data
if gene_expr:
    X_train = np.where(X_train == 0, 0.01, X_train)
    X_train = np.log2(X_train)

#standard scaling
scaler = StandardScaler()
scaler.fit(X_train) 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# Build the model
model = nn.Sequential(
    nn.Linear(feat, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Loss function and optimiser
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training parameters
num_epochs = 100
batch_size = 64
train_losses = []

for epoch in range(num_epochs):
    train_epoch_loss = 0.0
    num_batches = len(X_train) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = np.array(X_train[start_idx:end_idx], dtype=np.float32)
        y_batch = np.array(y_train[start_idx:end_idx], dtype=np.float32)

        optimizer.zero_grad()
        outputs = model(torch.tensor(X_batch))
        loss = criterion(outputs, torch.tensor(y_batch))

        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()

    avg_train_loss = train_epoch_loss / num_batches
    train_losses.append(avg_train_loss)


# predict the test data without calculating gradients
with torch.no_grad():
    test_inputs = torch.tensor(np.array(X_test), dtype=torch.float32)
    test_targets = torch.tensor(np.array(y_test), dtype=torch.float32)
    train_inputs = torch.tensor(np.array(X_train), dtype=  torch.float32)
    train_targets = torch.tensor(np.array(y_train), dtype=  torch.float32)
    test_outputs = model(test_inputs)
    train_outputs = model(train_inputs)
    #Calculate the accuracy
    predicted_labels = (test_outputs > 0.5).float()  # if output >0.5
    accuracy = accuracy_score(test_targets, predicted_labels)
    predicted_train_labels = (train_outputs  > 0.5).float()
    accuracy_train = accuracy_score(train_targets, predicted_train_labels )

print(f"test accuracy：{accuracy:.4f}")
print(f"train accuracy：{accuracy_train:.4f}")



end = time.time() #for timing the code

print('time elapsed: ', end-start)

# visualisation
plt.plot(train_losses)
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title('Training loss over time')
plt.show()



#Testing on the mystery data
#import mystery data
mystery_data_dna = pd.read_csv(r'./mystery_dna-meth_selected_features.csv')
mystery_data_gene = pd.read_csv(r'./mystery_gene-expr_selected_features.csv')
mystery_data_dna_full = pd.read_csv(r'./mystery_dna-meth.csv')
mystery_data_gene_full = pd.read_csv(r'./mystery_gene-expr.csv')




if gene_expr:
    if full: 
         mystery_filtered = mystery_data_gene_full[gene_data.columns]
         data = mystery_filtered
    else: 
        data = mystery_data_gene

else:
    if full:
        mystery_filtered = mystery_data_dna_full[dna_meth_data.columns]
        data = mystery_filtered
    else:
        data = mystery_data_dna



X = data.iloc[:, 2:]
y = data.iloc[:, 1]
y = pd.DataFrame(y)
y = y.replace({'Primary Tumor': 1, 'Solid Tissue Normal': 2})

X_train, X_mistery_test, y_train, y_mistery_test= train_test_split(X, y, test_size=0.3, random_state=42)

# print(X_mistery_test)
test_mystery_inputs = torch.tensor(np.array(X_mistery_test), dtype=torch.float32)
test_mystery_targets = torch.tensor(np.array(y_mistery_test), dtype=torch.float32)
test_outputs = model(test_mystery_inputs)
predicted_mys_labels = (test_outputs > 0.5).float()  # if output >0.5
accuracy_mystery = accuracy_score(test_mystery_targets , predicted_mys_labels )

print(f"mystery accuracy：{accuracy_mystery :.4f}")
