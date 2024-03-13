from DataSet import DataSet
from Model import Model
from torch.utils.data import random_split, DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim, max, sum

# 0 Load Data

data_set = DataSet("Dataset/Data_for_UCI_named.csv")

# 1 Split Data

train_dataset, validation_dataset, test_dataset = random_split(data_set, (7000, 1000, 2000))

# 2 Set Data Loader

train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
validation_loader = DataLoader(validation_dataset, len(validation_dataset), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

# 4 initial Model

model = Model()

# 5 Choice Loss & Optimizer Function

loss_function = CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 6 Train Model

for num in range(500):
    for data_batch, label_batch in train_loader:
        optimizer.zero_grad()

        out = model(data_batch)

        loss = loss_function(out, label_batch)

        loss.backward()

        optimizer.step()

    if (num + 1) % 50 == 0:
        for data_batch, label_batch in validation_loader:
            out = model(data_batch)
            predicted = max(out.data, 1)
            result = int(100 * sum(label_batch == predicted[1]) / len(validation_dataset))
            print(f"validation {num + 1} : {result}")
        if result > 95:
            break

# 7 Test Model

for data_batch, label_batch in test_loader:
    out = model(data_batch)
    predicted = max(out.data, 1)
    result = int(100 * sum(label_batch == predicted[1]) / len(test_dataset))
    print(f"test : {result}")