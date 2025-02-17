from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils import check_duplicated, check_null_values, extractCategorical, extractNumerical, impute_numerical, impute_categorical, one_hot_encoding
import pickle
import redis

app = Flask(__name__)

# Connect to Redis for caching
cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

@app.route("/")
def home():
    return "Welcome to the K-NN Model API!"

@app.route('/upload', methods=['POST'])
def upload_dataset():
    cache.delete('dataset')
    # Check if file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Get the uploaded file
    file = request.files['file']

    # Read the file into a Pandas DataFrame
    try:
        df = pd.read_csv(file)
        cache.set('dataset', df.to_json())
        # Check for duplicated rows
        duplicated_rows = check_duplicated(df)

        # Check for columns with null values
        columns_with_null = check_null_values(df, df.columns.to_list())

        return jsonify({
            'message': 'Dataset uploaded successfully',
            'no_of_duplicated_rows': int(duplicated_rows), # convert to int for JSON serialization
            'columns_with_null_values': columns_with_null
        })
    
    except Exception as e:
        #return jsonify({'error': 'Invalid file format'}), 400
        return jsonify({'error': str(e)}), 400
    


@app.route('/train', methods=['POST'])
def train_model():
    # Get user inputs from the form
    #data = request.form
    user_inputs = request.json
    n_neighbors = user_inputs.get('n_neighbors', 3)
    metric = user_inputs.get('metric', 'euclidean')
    weights = user_inputs.get('weights', 'uniform')
    test_size = user_inputs.get('test_size', 0.2)

    # Load dataset from Redis cache
    dataset = cache.get('dataset')
    if dataset is None:
        return jsonify({'error': 'No dataset uploaded'}), 400
    df = pd.read_json(dataset)

    #features = data.getlist('features')
    #features = data['features'].split(',') # Split comma-separated string into a list
    
    # Get features and target from user input
    features = user_inputs.get('features', df.columns[:-1].tolist())
    #target = data['target'] # Target variable
    target = user_inputs.get('target', df.columns[-1])
    print(target)

    #n_neighbors = int(data['n_neighbors']) # Number of neighbors

    #if not target:
    #    return jsonify({'error': 'Please select target'}), 400

    # Validate features and target

    #if not features:
    #    features = [col for col in df.columns if col != target]
    
    # Preprocess data
    try:
        X = df[features]
        y = df[target]
        numerical_cols = user_inputs.get('numerical_columns', X.select_dtypes(include=['int64', 'float64']).columns.tolist())
        categorical_cols = user_inputs.get('categorical_columns',X.select_dtypes(include=['object', 'category']).columns.tolist())
        num_impute_strategy = user_inputs.get('num_impute', 'mean')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=42)
        
        X_train_numerical = extractNumerical(data=X_train, numerical_cols=numerical_cols)
        X_train_categorical = extractCategorical(data=X_train, categorical_cols=categorical_cols)
        
        X_train_numerical, imputer_numerical = impute_numerical(X_train_numerical, num_impute_strategy)

        X_train_categorical = impute_categorical(X_train_categorical)
        
        X_train_cat_ohe = one_hot_encoding(X_train_categorical)[0]

        X_train_concat = pd.concat([X_train_numerical, X_train_cat_ohe], axis=1)

        # Standardize features
        standardizer = StandardScaler()
        standardizer.fit(X_train_concat)

        X_train_standardized_raw =  standardizer.transform(X_train_concat)
        X_train_standardized = pd.DataFrame(X_train_standardized_raw, 
                                            columns=X_train_concat.columns, 
                                            index=X_train_concat.index)

        y_train_flattened = y_train.values.ravel()
        
        # Train K-NN model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_standardized, y_train_flattened)

        # Evaluate the model
        X_test_numerical = extractNumerical(data=X_test, numerical_cols=numerical_cols)
        X_test_categorical = extractCategorical(data=X_test, categorical_cols=categorical_cols)

        X_test_numerical_imputed_raw = imputer_numerical.transform(X_test_numerical)
        X_test_numerical_imputed = pd.DataFrame(X_test_numerical_imputed_raw, columns=X_test_numerical.columns, index=X_test_numerical.index)

        X_test_categorical_imputed = impute_categorical(X_test_categorical)
        X_test_categorical_ohe = one_hot_encoding(X_test_categorical_imputed)[0]

        X_test_concat = pd.concat([X_test_numerical_imputed, X_test_categorical_ohe], axis=1)

        X_test_standardized_raw = standardizer.transform(X_test_concat)
        X_test_standardized = pd.DataFrame(X_test_standardized_raw, columns=X_test_concat.columns, index=X_test_concat.index)

        y_pred = knn.predict(X_test_standardized)
        accuracy = accuracy_score(y_test, y_pred)

        # Save model to Redis
        cache.set('knn_model', pickle.dumps(knn))
        cache.set('standardizer', pickle.dumps(standardizer))

        # return accuracy
        return jsonify({'message':'KNN model trained successfully',
                        'accuracy': accuracy,
                        'parameters':{
                            'n_neighbors':n_neighbors,
                            'metric': metric,
                            'weights': weights
                        }})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)



