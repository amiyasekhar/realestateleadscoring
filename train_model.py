import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set seed
np.random.seed(42)

# Generate synthetic lead data with BETTER SIGNAL (10000 leads)
n_samples = 10000

# Start with independent variables
age = np.random.randint(22, 70, n_samples)
income = np.random.randint(30000, 500000, n_samples)
credit_score = np.random.randint(300, 850, n_samples)

# Generate engagement based on income/credit (richer people engage more)
income_percentile = income / 500000
credit_percentile = credit_score / 850
base_engagement = (income_percentile + credit_percentile) / 2

website_visits = np.random.poisson(30 * base_engagement, n_samples)
email_opens = np.random.poisson(20 * base_engagement, n_samples)
property_views = np.random.poisson(15 * base_engagement, n_samples)

contact_attempts = np.random.poisson(5 * base_engagement, n_samples)
time_on_site_minutes = np.random.poisson(60 * base_engagement, n_samples)

# Intent signals correlated with income/credit
inquiry_submitted = (np.random.random(n_samples) < (0.2 + 0.3 * credit_percentile)).astype(int)
financing_pre_approved = (np.random.random(n_samples) < (0.3 + 0.3 * credit_percentile)).astype(int)
budget_disclosed = (np.random.random(n_samples) < (0.3 + 0.2 * income_percentile)).astype(int)
location_preference_set = (np.random.random(n_samples) < (0.4 + 0.2 * base_engagement)).astype(int)

data = {
    'age': age,
    'income': income,
    'credit_score': credit_score,
    'website_visits': website_visits,
    'email_opens': email_opens,
    'property_views': property_views,
    'contact_attempts': contact_attempts,
    'time_on_site_minutes': time_on_site_minutes,
    'inquiry_submitted': inquiry_submitted,
    'financing_pre_approved': financing_pre_approved,
    'budget_disclosed': budget_disclosed,
    'location_preference_set': location_preference_set,
}

df = pd.DataFrame(data)

# Create lead scores (IMPROVED - less randomness)
def create_lead_score(row):
    score = 0
    
    # Income (strong signal)
    if row['income'] > 200000:
        score += 4
    elif row['income'] > 150000:
        score += 3
    elif row['income'] > 80000:
        score += 2
    else:
        score += 1
    
    # Credit (strong signal)
    if row['credit_score'] > 780:
        score += 4
    elif row['credit_score'] > 750:
        score += 3
    elif row['credit_score'] > 650:
        score += 2
    else:
        score += 1
    
    # Engagement (medium signal)
    engagement = row['website_visits'] + row['email_opens'] + row['property_views'] * 2
    if engagement > 60:
        score += 3
    elif engagement > 30:
        score += 2
    else:
        score += 1
    
    # Intent (strong signal)
    if row['inquiry_submitted'] == 1:
        score += 2
    if row['financing_pre_approved'] == 1:
        score += 2
    if row['budget_disclosed'] == 1:
        score += 1
    if row['location_preference_set'] == 1:
        score += 1
    
    # Reduced randomness (only ±1 instead of ±2 to ±3)
    score += np.random.randint(-1, 2)
    
    if score >= 14:
        return 'High'
    elif score >= 9:
        return 'Medium'
    else:
        return 'Low'

df['lead_score'] = df.apply(create_lead_score, axis=1)

print(f"Total leads: {len(df)}")
print(f"\nLead Score Distribution:")
print(df['lead_score'].value_counts())

# Save dataset
df.to_csv('lead_scoring_dataset.csv', index=False)
print("\n✅ Dataset saved")

# Prepare data
X = df.drop('lead_score', axis=1)
y = df['lead_score']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance classes
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"Before SMOTE: {np.bincount(y_train)}")
print(f"After SMOTE: {np.bincount(y_train_balanced)}")

# Train XGBoost model
print("\nTraining XGBoost model with balanced data...")

sample_weights = compute_sample_weight('balanced', y_train_balanced)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train_balanced, 
    y_train_balanced,
    sample_weight=sample_weights
)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ XGBoost Model trained")
print(f"Accuracy: {accuracy:.2%}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Features:")
print(feature_importance.head().to_string(index=False))

# Save model
model_artifacts = {
    'model': model,
    'scaler': scaler,
    'label_encoder': le,
    'feature_names': list(X.columns)
}

with open('lead_scoring_model_xgb.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("\n✅ Model saved to 'lead_scoring_model_xgb.pkl'")
