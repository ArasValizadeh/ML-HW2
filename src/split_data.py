from sklearn.model_selection import train_test_split
from .config import TEST_SIZE, RANDOM_STATE

def split_data(df, target="bmd", test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X,y,test_size = TEST_SIZE , random_state = RANDOM_STATE)