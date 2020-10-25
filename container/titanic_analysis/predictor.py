import os
import json
import joblib
import flask
import pandas as pd
import sklearn.preprocessing as preprocessing

# The flask app for serving predictions
app = flask.Flask(__name__)

#Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# Load the model components
rfr = joblib.load(os.path.join(model_path, 'RFR.pkl'))
clf = joblib.load(os.path.join(model_path, 'LR.pkl'))

app.run(host='0.0.0.0', port=8080, debug=True)

@app.route("/")
def home():
    html = f"<h3>titanic survival prediction home</h3>"
    return html.format(format)

# check if the server is running and healthy
@app.route('/ping',methods=['GET'])
def ping():
    try:
        clf
        status = 200
    except:
        status = 400
    return flask.Response(response= json.dumps(' '),status=status, mimetype='application/json')

#perform prediction
@app.route('/invocations', methods=['POST'])
def predict():
    '''
    input looks like:
    {
        "PassengerId": "1",
        "Pclass": "3",
        "Name": "Kelly,Mr.James",
        "Sex": "male",
        "Age": "34",
        "SibSp": "0",
        "Parch": "0",
        "Ticket": "330911",
        "Fare": "7",
        "Cabin": "B45",
        "Embarked": "Q"
    }
    '''
    json_payload = flask.request.json
    inference_payload = pd.DataFrame(json_payload)
    processed_payload = process(inference_payload)
    prediction = list(clf.predict(processed_payload))

    return flask.jsonify({'prediction': prediction})

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def process(payload):
    payload.loc[ (payload.Fare.isnull()),'Fare'] = 0
    tmp_df = payload[['Age','Fare','Parch','SibSp','Pclass']]
    null_age = tmp_df[payload.Age.isnull()].to_numpy()

    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    payload.loc[ (payload.Age.isnull()),'Age'] = predictedAges

    payload = set_Cabin_type(payload)
    dummies_Cabin = pd.get_dummies(payload['Cabin'], prefix = 'Cabin')
    dummies_Embarked = pd.get_dummies(payload['Embarked'], prefix = 'Embarked')
    dummies_Sex = pd.get_dummies(payload['Sex'], prefix = 'Sex')
    dummies_Pclass = pd.get_dummies(payload['Pclass'], prefix = 'Pclass')

    df_test = pd.concat([payload, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)
    df_test.drop(['Pclass','Name',"Sex",'Ticket','Cabin','Embarked'], axis = 1, inplace = True)
    
    scaler = preprocessing.StandardScaler()
    df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1))
    df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1))
    
    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    return test

