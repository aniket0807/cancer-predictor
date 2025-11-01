import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as _pickle


# expose same top-level attributes used commonly
Load = _pickle.loads
Dump = _pickle.dumps
Unpickler = _pickle.Unpickler
Pickler = _pickle.Pickler

# also provide the common names
loads = _pickle.loads
dumps = _pickle.dumps
dump = _pickle.dump
load = _pickle.load
ProtocolVersion = getattr(_pickle, "HIGHEST_PROTOCOL", None)


def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # spliting data for testing and traning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data():
    data = pd.read_csv("data/data.csv")

    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        _pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        _pickle.dump(scaler, f)


if __name__ == '__main__':
    main()