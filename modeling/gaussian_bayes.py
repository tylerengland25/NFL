import sys
sys.path.append('c:\\Users\\tyler\\OneDrive\\Documents\\Python\\NFL')
from backend.preprocess.preprocess import main as load_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.exceptions import DataConversionWarning
import warnings


def gb():
    """
    Function:
        Create a GaussianNB using preprocessed data to determine the winner of a game.
        Standardizes data using StandardScaler.
        Selects features using ....
    
    Input:
        None
    
    Output:
        None
    """
    # Load data
    df = load_data()
    X = df.drop(['y'], axis=1)
    y = df[['y']]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

    # Pipeline
    pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('feature_selection', SelectPercentile(score_func=mutual_info_classif, percentile=10)),
            ('gb', GaussianNB())
        ]
    )

    # Fit and Score
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f'Accuracy: {round(accuracy_score(y_test, y_pred) * 100)}%')
    print(classification_report(y_test, y_pred))




if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    gb()