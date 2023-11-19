from pypy import app
from flask import render_template, request, url_for
from matplotlib import pyplot as plt
import mpld3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
classes = iris.target_names.tolist()

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/classifier_chooser', methods=['POST', 'GET'])
def classifier_chooser():
    _parameter_fields = ""
    classifier = request.form.get('classifiers')

    _parameter_fields = render_template(f"./classifiers/{classifier}.html")

    return render_template('home.html', parameter_fields=_parameter_fields)


@app.route('/classify_knn', methods=['POST'])
def classify_knn():
    _n_neighbors = request.form.get('n_neighbors', default=5) or 5
    _weights = request.form.get('weights', default='uniform') or 'uniform'
    _algorithm = request.form.get('algorithm', default='auto') or 'auto'
    _leaf_size = request.form.get('leaf_size', default=30) or 30

    clf = KNeighborsClassifier(n_neighbors=int(_n_neighbors), weights=_weights, algorithm=_algorithm, leaf_size=int(_leaf_size))
    _results = classify(clf)

    return render_template('home.html', results=_results)

@app.route('/classify_svc', methods=['POST'])
def classify_svc():
    _c = request.form.get('c', default=1.0) or 1.0
    _kernel = request.form.get('kernel', default='rbf') or 'rbf'

    clf = SVC(C=float(_c), kernel=_kernel)
    _results = classify(clf)

    return render_template('home.html', results=_results)

@app.route('/classify_mlp', methods=['POST'])
def classify_mlp():
    _hidden_layer_sizes = request.form.get('hidden_layer_sizes', default=100) or 100
    _activation = request.form.get('activation', default='relu') or 'relu'
    _learning_rate = request.form.get('learning_rate', default='constant') or 'constant'
    _max_iter = request.form.get('max_iter', default=200) or 200

    clf = MLPClassifier(hidden_layer_sizes=int(_hidden_layer_sizes), activation=_activation, learning_rate=_learning_rate, max_iter=int(_max_iter))
    _results = classify(clf)

    return render_template('home.html', results=_results)

@app.route('/classify_dt', methods=['POST'])
def classify_dt():
    _criterion = request.form.get('criterion', default='gini') or 'gini'
    _splitter = request.form.get('splitter', default='best') or 'best'
    _max_depth = request.form.get('max_depth', default=None) or None
    if _max_depth is not None:
        _max_depth = int(_max_depth)

    clf = DecisionTreeClassifier(criterion=_criterion, splitter=_splitter, max_depth=_max_depth)
    _results = classify(clf)

    return render_template('home.html', results=_results)

@app.route('/classify_rf', methods=['POST'])
def classify_rf():
    _n_estimators = request.form.get('n_estimators', default=100) or 100
    _criterion = request.form.get('criterion', default='gini') or 'gini'
    _max_depth = request.form.get('max_depth', default=2) or 2

    clf = RandomForestClassifier(n_estimators=_n_estimators, criterion=_criterion, max_depth=_max_depth)
    _results = classify(clf)

    return render_template('home.html', results=_results)

def classify(clf):
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    _accuracy = accuracy_score(y_test, y_pred)
    _f1_score = f1_score(y_test, y_pred, average='macro')

    cm = confusion_matrix(y_test, y_pred)
    _conf_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    _conf_matrix.plot(ax=ax, values_format='.0f')
    _conf_matrix_html = mpld3.fig_to_html(fig)

    return render_template('results.html', classifier='K Nearest Neighbors', accuracy=_accuracy, f1_score=_f1_score,
                               conf_matrix=_conf_matrix_html)