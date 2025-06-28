from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess(final):
    final['reordered']=final['up_reorder_count'].apply(lambda x: 1 if x > 0 else 0)
    x= final.drop(columns=['user_id', 'product_id', 'reordered'])
    y = final['reordered']

    scaler= StandardScaler()    
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Save the scaler
    joblib.dump(scaler, 'models/scaler.save')

