import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
sns.set()

country_df = pd.read_csv('./data/Country-data.csv')

st.header('Clustering Countries')
st.subheader('Clustering countries bases on differente indicators')

st.dataframe(country_df.head())

@st.cache
def processing_info():
    only_features_df = country_df.drop('country', axis=1)
    only_features_df.head()
    sc = StandardScaler()
    scaled_features = sc.fit_transform(only_features_df)
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(scaled_features)
                for k in range(1, 10)]
    inertias = [model.inertia_ for model in kmeans_per_k]
    silhouette_scores = [silhouette_score(scaled_features, model.labels_)
                     for model in kmeans_per_k[1:]]
    return (inertias, silhouette_scores, scaled_features, sc)

(inertias, silhouette_scores, scaled_features, sc) = processing_info()

fig1, ax1 = plt.subplots()
ax1 = sns.scatterplot(x = range(1, 10), y = inertias)
ax1 = sns.lineplot(x = range(1, 10), y = inertias)
ax1.set_xlabel("Number of clusters", fontsize=14)
ax1.set_ylabel("Inertia", fontsize=14)  
st.pyplot(fig1, clear_figure=True)

fig2, ax2 = plt.subplots()
ax2 = sns.scatterplot(x = range(2, 10), y = silhouette_scores)
ax2 = sns.lineplot(x = range(2, 10), y = silhouette_scores)
ax2.set_xlabel("Number of clusters", fontsize=14)
ax2.set_ylabel("Silhouette score", fontsize=14)  
st.pyplot(fig2, clear_figure=True)

st.subheader('''3 Clusters are convenient according to the graphs
                - Developed Country
                - Developing Country
                - Underdeveloped Country
''')

clusters = KMeans(n_clusters=3, random_state=42).fit(scaled_features)

child_mortality = st.sidebar.slider('Child Mortality', min_value=2.6, max_value=208.0)
exports = st.sidebar.slider('Exports', min_value=0.109, max_value=200.0)
health = st.sidebar.slider('Health', min_value=1.81, max_value=17.9)
imports = st.sidebar.slider('Imports', min_value=0.0659, max_value=174.0)
income = st.sidebar.slider('Income', min_value=609.0, max_value=125000.0)
inflation = st.sidebar.slider('Inflation', min_value=-4.21, max_value=104.0)
life_expec = st.sidebar.slider('Life Expectancy', min_value=32.1, max_value=82.8)
total_fer = st.sidebar.slider('Total Fertility', min_value=1.15, max_value=7.49)
gdpp = st.sidebar.slider('GDP per Capita', min_value=231.0, max_value=105000.0)

input_scaled = sc.transform([[child_mortality,exports,health,imports,income,inflation,life_expec,total_fer,gdpp]])
predicted_label = clusters.predict(input_scaled)

st.subheader('''Choose different values on the bar on the left to see to which cluster it will belong
''')

if (predicted_label == 0):
    st.subheader('Developed Country')
elif (predicted_label == 1):
    st.subheader('Underdeveloped Country')
else:
    st.subheader('Developing Country')

st.subheader('Full notebook in the following link https://www.kaggle.com/johnsnow27/kmeans-silhouette-score')
