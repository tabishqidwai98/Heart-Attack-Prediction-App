import streamlit as st
import numpy as np
import pandas as pd
import joblib


st.title("Heart attack prediction App")

st.sidebar.image("py.jpg")
sidebar = st.sidebar.selectbox(
    "The app features",
    ("Main Page", "Dataset", "Analysis", "lets see possibilities")
)

if sidebar == "Main Page":
    st.header("The Heart Disease")
    
    st.write("""A heart attack, also called a myocardial infarction, happens when a part of the heart muscle doesn't get enough blood.

The more time that passes without treatment to restore blood flow, the greater the damage to the heart muscle.

Coronary artery disease (CAD) is the main cause of heart attack. A less common cause is a severe spasm, or sudden contraction, of a coronary artery that can stop blood flow to the heart muscle.""")

    st.image("ty.jpg")
    st.subheader("Symptoms")

    st.write("""
            The major symptoms of a heart attack are

- Chest pain or discomfort. Most heart attacks involve discomfort in the center or left side of the chest that lasts for more than a few minutes or that goes away and comes back. The discomfort can feel like uncomfortable pressure, squeezing, fullness, or pain.
- Feeling weak, light-headed, or faint. You may also break out into a cold sweat.
- Pain or discomfort in the jaw, neck, or back.
- Pain or discomfort in one or both arms or shoulders.
- Shortness of breath. This often comes along with chest discomfort, but shortness of breath also can happen before chest discomfort.
""")
    
    st.subheader("Risk factors")

    st.write("""Several health conditions, your lifestyle, and your age and family history can increase your risk for heart disease and heart attack. These are called risk factors. About half of all Americans have at least one of the three key risk factors for heart disease: high blood pressure, high blood cholesterol, and smoking.2

Some risk factors cannot be controlled, such as your age or family history. But you can take steps to lower your risk by changing the factors you can control.
""")
    
    st.subheader("Recover after a heart attack")

    st.write("""
        If you’ve had a heart attack, your heart may be damaged. This could affect your heart’s rhythm and its ability to pump blood to the rest of the body. You may also be at risk for another heart attack or conditions such as stroke, kidney disorders, and peripheral arterial disease (PAD).

You can lower your chances of having future health problems following a heart attack with these steps:

- Physical activity—Talk with your health care team about the things you do each day in your life and work. Your doctor may want you to limit work, travel, or sexual activity for some time after a heart attack.
- Lifestyle changes—Eating a healthier diet, increasing physical activity, quitting smoking, and managing stress—in addition to taking prescribed medicines—can help improve your heart health and quality of life. Ask your health care team about attending a program called cardiac rehabilitation to help you make these lifestyle changes.
- Cardiac rehabilitation—Cardiac rehabilitation is an important program for anyone recovering from a heart attack, heart failure, or other heart problem that required surgery or medical care. Cardiac rehab is a supervised program that includes
1. Physical activity
2. Education about healthy living, including healthy eating, taking medicine as prescribed, and ways to help you quit smoking
3. Counseling to find ways to relieve stress and improve mental health
             
A team of people may help you through cardiac rehab, including your health care team, exercise and nutrition specialists, physical therapists, and counselors or mental health professionals.


""")


if sidebar == "Dataset":
    st.write("Here's the dataset")

    df = pd.read_csv("C:/Users/tabis/Documents/GitHub/Heart Attack App/Heart datasets/heart.csv")

    st.table(df)

    lst = ["Main Page", "Dataset", "Analysis", "lets see possibilities"]

    st.line_chart(df)

#st.sidebar() 

if sidebar == "Analysis":

    st.header("Analysis")
    st.write("Insights dataset")

    st.image("img/output.png")
    st.image("img/output1.png")
    st.image("img/output2.png")
    st.image("img/3.png")
    st.image("img/output4.png")
    st.image("img/5.png")
    st.image("img/6.png")
    st.image("img/7.png")

    st.image("img/8.png")
    st.image("img/9.png")
    st.image("img/10.png")
    st.image("img/11.png")

if sidebar == "lets see possibilities":
    st.header("lets use AI")
    st.write("*Lets see what does it say about heart*")
    st.subheader("Your details")

    sex = st.selectbox('Gender = Male : 1,  Female : 0', (0, 1))
    cp = st.number_input("Chest Pain. Are u feeling any Chest pain, if yes please enter the chest pain(cp) prescribed by the doctor", min_value=0, max_value=3)
    exng = st.selectbox("Exercise induced angina. Is there any pain while working out, doing a simple exercise or while in stress", (0, 1))
    oldpeak = st.number_input("Previous peak, Values may be in between 0 and 6.2 but these estimated values might differ", step=0.1, format="%.2f")
    caa = st.number_input("Number of major vessels having issue", min_value=0, max_value=4)

    clicked = st.button("Predict")

    if clicked:
        try:
            model = joblib.load(open('C:/Users/tabis/Documents/GitHub/heart attack prediction app/model.pkl', 'rb'))

            if model != None:
                # Ensure the input features are in the correct format
                features = np.array([[sex, cp, exng, oldpeak, caa]])
                predicted = model.predict(features)
                
                st.header("Predicted Result")
                st.info('0 (No possibility of heart attack), 1 (Future heart attack detected)')
                st.success(predicted[0])

        except Exception as e:
            st.error(f"Error loading the model: {e}")