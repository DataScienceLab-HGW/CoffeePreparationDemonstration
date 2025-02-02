import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from random import randint
import datetime as dt
import socket
from struct import unpack
from StatePredictionModel import PredictionModel



simulation = False
debug = False
weights = pd.DataFrame(columns=['Zeitstempel','Kaffeebehälter', 'French Press', 'Wasserkocher'])
sum_coffeebox = 0
sum_frenchpress = 0
sum_boiler = 0

modell_text = ""

coffee_in_place = True
frenchpress_in_place = True
boiler_in_place = True
boiler_full_water = False
powerPress_on = False

coffee_model = PredictionModel()


st.set_page_config(
    page_title="Kaffeemodell",
    page_icon="☕",
    layout="wide",
)

if not simulation:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the port
    host, port = '10.130.5.106', 65000
    server_address = (host, port)

    print(f'Starting UDP server on {host} port {port}')
    sock.bind(server_address)


def compute_data() -> pd.DataFrame:
    global sum_coffeebox, sum_frenchpress, sum_boiler, weights
    if simulation:
        sum_coffeebox, sum_frenchpress, sum_boiler = randint(28,35), randint(28,35), randint(28,35)
        weights.loc[len(weights.index)] = [dt.datetime.now().strftime('%H:%M:%S'), sum_coffeebox, sum_frenchpress, sum_boiler]
    
    else:
        message, address = sock.recvfrom(4096)
        sum_coffeebox, sum_frenchpress, sum_boiler = unpack('3f', message)
        sum_coffeebox = sum_coffeebox//100
        sum_frenchpress = sum_frenchpress//100
        sum_boiler = sum_boiler//100
        # update df
        weights.loc[len(weights.index)] = [dt.datetime.now().strftime('%H:%M:%S'), sum_coffeebox, sum_frenchpress, sum_boiler]

modell_states = ["1110"]
text = "Bitte nehmen Sie den Wasserkocher"
statesString = "Start"
def update_state():
    global text, statesString
    coffee_in_place, frenchpress_in_place, boiler_in_place, boiler_full_water = True, True, True, False
    if sum_coffeebox > 300:
        coffee_in_place = True
    else:
        coffee_in_place = False

    if sum_frenchpress > 300 and sum_frenchpress <= 1500:
        frenchpress_in_place = True
        powerPress_on = False
    elif sum_frenchpress > 1500:
        frenchpress_in_place = True
        powerPress_on = True
    else:
        frenchpress_in_place = False

    if sum_boiler > 300 and sum_boiler <= 500:
        boiler_in_place = True
        boiler_full_water = False
    elif sum_boiler > 300:
        boiler_full_water = True
        boiler_in_place = True
    else:
        boiler_in_place = False
        boiler_full_water = False
    
    old_statesString = statesString
    statesString = str(int(coffee_in_place)) + "" + str(int(frenchpress_in_place)) + "" + str(int(boiler_in_place))  + "" + str(int(boiler_full_water)) 
    if old_statesString == "Start":
        text = "Bitte nehmen Sie den Wasserkocher"
        return text

    text_old = text
    if statesString == old_statesString:
        if modell_states[-1] != statesString:
            print(modell_states)
            text = coffee_model.make_prediction_on_observation(observation_as_vector=statesString)
            modell_states.append(statesString)
            # with open("modell_states.txt", "w") as f:
            #     for state in modell_states:
            #         f.write(state + "\n")
            #     f.close()
            print(statesString)
            if text.__contains__("Nach dem Kochen des Wassers, Wasserkocher nehmen"):
                text = "Gießen Sie das Wasser in die French Press und stellen Sie den Wasserkocher wieder ab"
            elif text.__contains__("Wasserkocher nehmen"):
                text = "Befüllen Sie den Wasserkocher mit ungefähr 1L Wasser und stellen Sie ihn wieder ab"
            elif text.__contains__("Wasserkocher befüllen"):
                text = "Schalten Sie den Wasserkocher an und nehmen Sie das Kaffeebehältnis"
            elif text.__contains__("Kaffebehälter nehmen"):
                text = "Geben Sie 6 Esslöffel Kaffee in die French Press und stellen Sie das Behältnis wieder ab"
            elif text.__contains__("Behälter abstellen"):
                text = "Wenn das Wasser gekocht hat, nehmen Sie den Wasserkocher"
            elif text.__contains__("Wasser in die French Press"):
                text = "Warten Sie ein paar Minuten, drücken Sie die French Press nach unten und nehmen Sie sie in die Hand"
            elif text.__contains__("French Press drücken"):
                text = "Nehmen Sie die French Press in die Hand"
            elif text.__contains__("French Press nehmen"):
                text = "Gießen Sie den Kaffee in eine Tasse und stellen Sie die French Press zurück an ihren Platz"
            elif text.__contains__("Kaffee servieren"):
                text = "Genießen Sie Ihren Kaffee"
            return text
    print("Wait for one more second")
    return text_old
    #return coffee_in_place, frenchpress_in_place, boiler_in_place, boiler_full_water

def update_display():
    print("TODO")


# dashboard title
st.title("Kaffeekochen mit KI-Assistenzmodell")

placeholder = st.empty()
states = ["1110"]
while True:
    
    #old_states = states
    compute_data()
    modell_text = update_state()
    # print(states)
    # if states == old_states:
    #     coffee_in_place, frenchpress_in_place, boiler_in_place, boiler_full_water = states
    # else: 
    #     print("Wait one more second")


    # if debug:
    #     modell_text = update_display_debug()
    # else:
    #     modell_text = coffee_model.make_prediction_on_observation(observation_as_vector=states)
    
    with placeholder.container():
        if len(weights.index) >= 30:
            last_weights = weights.tail(30)
        else:
            last_weights = weights
        # create two columns for charts
        [fig_col1] = st.columns(1)
        modell, plots = st.tabs(["Modell", "Grafiken"])
        with modell:
            st.header("Kaffeemodell")
            st.write("##")
            st.write("##")
            st.write("##")
            st.markdown(""" <div>
                                <h1 
                                    style=
                                    'text-align: center;
                                    border-style: solid;
                                    background-color: #444;'
                                > 
                                    {} 
                                </h1>
                            </div>
                        """.format(modell_text), unsafe_allow_html=True)

        with plots:
            st.markdown("### Sensor-Messungen")
            fig = px.line(
                data_frame=last_weights, y=last_weights.columns[1:], x="Zeitstempel"
            )
            st.write(fig)
            st.markdown("### Rohdatenansicht")
            st.dataframe(weights)
            time.sleep(1)