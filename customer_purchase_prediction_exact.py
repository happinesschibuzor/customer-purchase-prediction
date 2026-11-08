# Customer Purchase Prediction
# RetailTech Solutions

# Task 1
import pandas as pd


df = pd.read_csv("raw_customer_data.csv")
df.head()

def clean_data(df):
    df = df.copy()

    # customer_id: unique integer, no missing values
    assert df['customer_id'].notna().all(), "customer_id has missing values"
    assert df['customer_id'].is_unique, "customer_id must be unique"
    df['customer_id'] = df['customer_id'].astype(int)

    # time_spent: float, missing replaced with median
    df['time_spent'] = df['time_spent'].astype(float)
    df['time_spent'] = df['time_spent'].fillna(df['time_spent'].median())

    # pages_viewed: integer, missing replaced with mean
    df['pages_viewed'] = df['pages_viewed'].fillna(df['pages_viewed'].mean())
    df['pages_viewed'] = df['pages_viewed'].round().astype(int)

    # basket_value: float, missing replaced with 0
    df['basket_value'] = df['basket_value'].astype(float).fillna(0)

    # device_type: Mobile, Desktop, Tablet, otherwise Unknown
    df['device_type'] = df['device_type'].fillna('Unknown')

    # customer_type: New or Returning, missing replaced with New
    df['customer_type'] = df['customer_type'].fillna('New')

    # purchase: binary target
    df['purchase'] = df['purchase'].astype(int)

    return df


# Task 2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

model_data = pd.read_csv("model_data.csv")
model_data.head()

model_features_set = model_data.copy()

# Scaling the numerical column from 0-1
numeric_col = ["time_spent", "pages_viewed", "basket_value"]
scaler = MinMaxScaler()
model_features_set[numeric_col] = scaler.fit_transform(model_features_set[numeric_col])

category_features =['device_type', 'customer_type']
model_features_set = pd.get_dummies(model_features_set, columns=category_features, dtype=int)

print(model_features_set.head())


# Task 3
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


torch.manual_seed(42)

train_df = pd.read_csv("input_model_features.csv")
val_df = pd.read_csv("validation_features.csv")

feature_col = [c for c in train_df.columns if c not in ('customer_id', 'purchase')]

# Hold out 20% of the labelled data to validate the model
X_tr, X_ho, y_tr, y_ho = train_test_split(
    train_df[feature_col].values,
    train_df['purchase'].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df['purchase']
)

X_train = torch.tensor(X_tr, dtype=torch.float32)
y_train = torch.tensor(y_tr, dtype=torch.float32).view(-1,1)
X_holdout = torch.tensor(X_ho, dtype=torch.float32)

print(X_train.shape)
print(y_train.shape)

# Training the model

purchase_model = nn.Sequential(
    nn.Linear(len(feature_col), 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(purchase_model.parameters(), lr=0.01)

# Training the loop

epochs = 200
for epoch in range(epochs):
    purchase_model.train()
    optimizer.zero_grad()
    outputs = purchase_model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


# Validate on the holdout set
purchase_model.eval()
with torch.no_grad():
    holdout_pred = (purchase_model(X_holdout).squeeze() >= 0.5).int().numpy()

print("Holdout accuracy:", round(accuracy_score(y_ho, holdout_pred), 4))
print("Holdout F1:", round(f1_score(y_ho, holdout_pred), 4))


# Predict on the validation set, using the same feature columns and order
X_val = torch.tensor(val_df[feature_col].values, dtype=torch.float32)

purchase_model.eval()

with torch.no_grad():
    probabilities = purchase_model(X_val).squeeze()

predicted_purchase = (probabilities >= 0.5).int().numpy()

validation_predictions = pd.DataFrame({
    'customer_id': val_df['customer_id'],
    'purchase': predicted_purchase
})

print(validation_predictions.head())
