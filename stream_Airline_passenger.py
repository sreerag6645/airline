 
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('Airline passenger')






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('airline sats.jfif')
    image_office = Image.open('side.jfif')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("single", "Batch"))
    st.sidebar.info('This app is created to prdicting Airline Passenger Satisfaction')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)
    st.title("Airline satisfaction")
    if add_selectbox == 'single':
        Age=st.number_input('Age' , min_value=7, max_value=85, value=7)
        Flight_Distance =st.number_input('Flight_Distance',min_value=31.0, max_value=4983.0, value=31.0)
        Infligh_wifi_service =st.number_input('Infligh_ wifi_service',min_value=0, max_value=5, value=0)
        Departure_Arrival_time_convenient = st.number_input('Departure_Arrival_time_convenient', min_value=0, max_value=5, value=0)
        Ease_of_Online_booking = st.number_input('Work_accidentEase_of_Online_booking',  min_value=0, max_value=5, value=0)
        Gate_location = st.number_input('Gate_location',  min_value=1, max_value=5, value=1)
        Food_and_drink = st.number_input('Food_and_drink',  min_value=0, max_value=5, value=0)
        Online_boarding = st.number_input('Online_boarding',  min_value=0, max_value=5, value=0)
        Seat_comfort = st.number_input('Seat_comfort',  min_value=0, max_value=5, value=0)
        Inflight_entertainment = st.number_input('Inflight_entertainment',  min_value=0, max_value=5, value=0)
        On_board_service= st.number_input('On_board_service',  min_value=0, max_value=5, value=0)
        Leg_room_ervice= st.number_input('Leg room service',  min_value=0, max_value=5, value=0)
        Baggage_handling = st.number_input('Baggage handling',  min_value=1, max_value=5, value=1)
        Checkin_service= st.number_input('Checkin service',  min_value=0, max_value=5, value=0)
        Inflight_service= st.number_input('Inflight service',  min_value=0, max_value=5, value=0)
        Cleanliness = st.number_input('Cleanliness',  min_value=0, max_value=5, value=0)
        Departure_Delay_in_Minutes= st.number_input('Departure Delay in Minutes',  min_value=0, max_value=1305, value=0)
        Arrival_Delay_in_Minutes= st.number_input('Arrival Delay in Minutes',  min_value=0, max_value=1280, value=0)
        Gender = st.selectbox('Gender', ['Female', 'Male'])
        Customer_Type = st.selectbox('Loyal Customer',	'disloyal Customer')
        Type_of_Travel = st.selectbox('Type of Travel', ['Business travel','Personal Travel'])
        Class= st.selectbox('Class', ['Business','Eco','Eco Plus', 'Bu'])
        output=""
        input_dict={'Gender':Gender, 'Customer Type':Customer_Type, 'Age':Age, 'Type_of_Travel':Type_of_Travel, 'Class':Class,
       'Flight_Distance':Flight_Distance, 'Infligh_ wifi_service':Infligh_wifi_service,
       'Departure_Arrival_time_convenient':Departure_Arrival_time_convenient, 'Ease_of_Online_booking':Ease_of_Online_booking,
       'Gate_location':Gate_location, 'Food_and_drink':Food_and_drink, 'Online_boarding':Online_boarding, 'Seat_comfort':Seat_comfort,
       'Inflight_entertainment':Inflight_entertainment, 'On_board_service':On_board_service, 'Leg_room _ervice':Leg_room_ervice  ,
       'Baggage_handling':Baggage_handling, 'Checkin_service':Checkin_service, 'Inflight_service':Inflight_service,
       'Cleanliness':Cleanliness, 'Departure_Delay_in_Minutes':Departure_Delay_in_Minutes, 'Arrival_Delay_in_Minutes':Arrival_Delay_in_Minutes}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)            
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
