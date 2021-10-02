#Import libraries 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go


#Define what the app does and style the layout of the app
st.title("Weight Prediction App")

st.write("This app is used to **predict the weight** of a person if the **person height** is known")


st.sidebar.header("Input Parameters")
height = st.sidebar.slider("Height of a person (m)", 0.3, 2.5, 1.7)

st.write("")
st.write("")
st.write("""\
The trend of height against the weight of people
""")

#Read the dataset and store in a variable
df_height_weight = pd.read_csv("data.csv")

df_height_weight.head()

#Train the model using Linear Regression Algorithm
endogenous_var = df_height_weight["Weight"]

exogenous_var = df_height_weight.drop("Weight", axis='columns')

x_train, x_test, y_train, y_test = train_test_split(exogenous_var, endogenous_var, 
random_state=10, train_size=0.8)

model = LinearRegression()

model.fit(x_train, y_train)

slope, intercept = model.coef_, model.intercept_


#Plot the trend of the heigh against the weight
x_values = np.linspace(x_train["Height"].min(), x_train["Height"].max())

model_func = lambda m,c,x: c + m*x

y_values = model_func(slope, intercept, x_values)

df = pd.DataFrame({"x-values":x_values, "y-values":y_values})

fig = go.Figure()

fig.add_trace(go.Line(x=df["x-values"], y=df["y-values"], name="Linear model"))
fig.add_trace(go.Scatter(x=df_height_weight["Height"], y=df_height_weight["Weight"],
name="Height and Weight", mode="markers"))

st.write("")
st.write(fig)

st.plotly_chart(fig)

with st.form("User Details"):
    firstname = st.text_input("First Name")
    surname = st.text_input("Surname")

    st.form_submit_button()


st.write("")
st.write("**Predict Your Weight**\n******************")
st.write(pd.DataFrame({"Height (m)":height, 
                        "Weight (Kg)": model.predict([[height]])}))



map_data = pd.DataFrame({"lat" : 4.42, "lon": 7.22}, index=[1,2])
st.map(map_data)

with st.form("Gender"):
    male = st.checkbox("Male")
    female = st.checkbox("Female")

    st.form_submit_button()


st.selectbox("Ice cream", options=[1,2])

left_column, right_column = st.columns(2)
pressed = left_column.button('Press me?')
if pressed:
  right_column.write("Woohoo!")

expander = st.expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")

st.radio("Gender", options=("Male", "Female"))
st.checkbox("Expresso")

