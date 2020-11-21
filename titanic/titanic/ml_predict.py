def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
    import pickle
    x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]
    random_forest = pickle.load(open("titanic_model.sav","rb"))
    predictions = random_forest.predict(x)
    if predictions == 0:
        prediction = 'Not survived'
    elif predictions == 1:
        prediction = 'Survived'
    else:
        prediction = 'Error'
    return prediction