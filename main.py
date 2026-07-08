import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles, make_moons, make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sample_size = st.sidebar.slider("Sample Size", 200, 1000, 200, 50)
data = st.sidebar.selectbox(
    "Sample Type",
    ["linear", "linear with noise", "circle", "moon", "blob"]
)

if data == 'moon' or data == 'circle':
    noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.1, 0.01)

if data == 'circle':
    factor = st.sidebar.slider("Factor", 0.0,0.99, 0.8,0.02)

if data == "linear":
    X, y = make_classification(
        n_samples=sample_size,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2,
        random_state=42
    )

elif data == "linear with noise":
    X, y = make_classification(
        n_samples=sample_size,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2,
        flip_y=0.15,
        random_state=42
    )

elif data == "circle":
    X, y = make_circles(
        n_samples=sample_size,
        noise=noise,
        factor=factor,
        random_state=42
    )

elif data == "moon":
    X, y = make_moons(
        n_samples=sample_size,
        noise=noise,
        random_state=42
    )

elif data == "blob":
    X, y = make_blobs(
        n_samples=sample_size,
        centers=3,
        cluster_std=1.2,
        random_state=42
    )

X, x_test, y, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

kernel = st.sidebar.selectbox("Kernel", ['linear','rbf','poly','sigmoid'])
C = st.sidebar.number_input("C",max_value=1000.0,min_value=0.01,step=0.5)

model = SVC(
    kernel=kernel,
    C=C
)

model.fit(X,y)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

st.metric("Training Accuracy", f"{accuracy:.3f}")

fig1, ax = plt.subplots()
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.set_axis_off()
ax.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")
ax.scatter(
    X[:,0],
    X[:,1],
    c=y,
    cmap="bwr", 
    marker = 'o'
)

st.pyplot(fig1, width = 500)
