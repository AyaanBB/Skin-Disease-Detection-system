
from kaggle import get_clean_dataframe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def processing():     
    df = get_clean_dataframe()

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['dx'])


    X = df['path']
    y = df['label']

    X_train,X_test,y_train,y_test = train_test_split (
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test = processing()
print(f"Training on: {len(X_train)} samples")
print(f"Testing on: {len(X_test)} samples")

