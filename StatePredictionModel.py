import numpy as np

ERR_IMPOSSIBLE_OBS = "The observation is not possible for the model. Please set all objects to their initial state"
ERR_STATE = "Error"
ERR_IMPOSSIBLE_STATE = "Impossible state. Please return back to the previous step: "
ERR_MESSAGE_BEGIN = "The following Errors occurred during execution:"

MOST_PROBABLE_STATE_MESSAGE = "Most probable state for observation"
PROBABILITY_MESSAGE = "Probability"

class PredictionModel:
    def __init__(self, states=None, emission_probabilities=None, possible_sensor_observations=None, start_probabilities=None, transition_probabilities=None):
      
        self.states = states if states else self.set_default_states()
        self.transition_probabilities = transition_probabilities if transition_probabilities else self.set_default_transition_probabilities()
        self.emission_probabilities = emission_probabilities if emission_probabilities else self.set_default_emission_probabilities()
        self.possible_sensor_observations = possible_sensor_observations if possible_sensor_observations else self.set_default_possible_sensor_observations()
        self.start_probabilities = start_probabilities if start_probabilities else self.set_default_start_probabilities()
        self.alpha_matrix = None
        self.last_state_prediction = "Initial state"
        self.last_state_prediction_probability = None

    def set_default_states(self):

        states = [
                        "Wasserkocher nehmen",
                        "Wasserkocher befüllen und wieder abstellen",
                        "Wasserkocher anmachen, Kaffebehälter nehmen",
                        "Kaffeepulver in die French Press geben und Behälter abstellen",
                        "Nach dem Kochen des Wassers, Wasserkocher nehmen",
                        "Wasser in die French Press geben und Wasserkocher abstellen",
                        "3 minuten warten, dann French Press drücken",
                        "French Press nehmen",
                        "Kaffee servieren und French Press abstellen",
                        "END"
        ]

        return states

    def set_default_transition_probabilities(self):

        transition_probabilities =  {

                                "Wasserkocher nehmen":{
                                    "Wasserkocher nehmen":0.0,
                                    "Wasserkocher befüllen und wieder abstellen":0.9,
                                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.0,
                                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.0,
                                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.0,
                                    "Wasser in die French Press geben und Wasserkocher abstellen":0.0,
                                    "3 minuten warten, dann French Press drücken":0.0,
                                    "French Press nehmen":0.0,
                                    "Kaffee servieren und French Press abstellen":0.0,
                                    "END":0.0
                                },

                                "Wasserkocher befüllen und wieder abstellen":{
                                    "Wasserkocher nehmen":0.0,
                                    "Wasserkocher befüllen und wieder abstellen":0.0,
                                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.9,
                                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.0,
                                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.0,
                                    "Wasser in die French Press geben und Wasserkocher abstellen":0.0,
                                    "3 minuten warten, dann French Press drücken":0.0,
                                    "French Press nehmen":0.1,
                                    "Kaffee servieren und French Press abstellen":0.0,
                                    "END":0.0
                                },

                                "Wasserkocher anmachen, Kaffebehälter nehmen":{
                                    "Wasserkocher nehmen":0.0,
                                    "Wasserkocher befüllen und wieder abstellen":0.0,
                                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.0,
                                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.9,
                                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.1,
                                    "Wasser in die French Press geben und Wasserkocher abstellen":0.0,
                                    "3 minuten warten, dann French Press drücken":0.0,
                                    "French Press nehmen":0.0,
                                    "Kaffee servieren und French Press abstellen":0.0,
                                    "END":0.0
                                },

                                "Kaffeepulver in die French Press geben und Behälter abstellen":{
                                    "Wasserkocher nehmen":0.0,
                                    "Wasserkocher befüllen und wieder abstellen":0.0,
                                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.0,
                                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.0,
                                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.9,
                                    "Wasser in die French Press geben und Wasserkocher abstellen":0.0,
                                    "3 minuten warten, dann French Press drücken":0.0,
                                    "French Press nehmen":0.1,
                                    "Kaffee servieren und French Press abstellen":0.0,
                                    "END":0.0
                                },

                                "Nach dem Kochen des Wassers, Wasserkocher nehmen":{
                                    "Wasserkocher nehmen":0.0,
                                    "Wasserkocher befüllen und wieder abstellen":0.0,
                                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.0,
                                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.0,
                                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.0,
                                    "Wasser in die French Press geben und Wasserkocher abstellen":0.9,
                                    "3 minuten warten, dann French Press drücken":0.0,
                                    "French Press nehmen":0.1,
                                    "Kaffee servieren und French Press abstellen":0.0,
                                    "END":0.0
                                },

                                "Wasser in die French Press geben und Wasserkocher abstellen":{
                                    "Wasserkocher nehmen":0.0,
                                    "Wasserkocher befüllen und wieder abstellen":0.0,
                                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.0,
                                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.0,
                                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.0,
                                    "Wasser in die French Press geben und Wasserkocher abstellen":0.05,
                                    "3 minuten warten, dann French Press drücken":0.9,
                                    "French Press nehmen":0.05,
                                    "Kaffee servieren und French Press abstellen":0.0,
                                    "END":0.0
                                },

                                "3 minuten warten, dann French Press drücken":{
                                    "Wasserkocher nehmen":0.0,
                                    "Wasserkocher befüllen und wieder abstellen":0.0,
                                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.0,
                                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.0,
                                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.0,
                                    "Wasser in die French Press geben und Wasserkocher abstellen":0.00,
                                    "3 minuten warten, dann French Press drücken":0.1,
                                    "French Press nehmen":0.9,
                                    "Kaffee servieren und French Press abstellen":0.0,
                                    "END":0.0
                                },

                                    "French Press nehmen":{
                                    "Wasserkocher nehmen":0.0,
                                    "Wasserkocher befüllen und wieder abstellen":0.0,
                                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.0,
                                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.0,
                                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.0,
                                    "Wasser in die French Press geben und Wasserkocher abstellen":0.00,
                                    "3 minuten warten, dann French Press drücken":0.0,
                                    "French Press nehmen":0.1,
                                    "Kaffee servieren und French Press abstellen":0.9,
                                    "END":0.0
                                },

                                    "Kaffee servieren und French Press abstellen":{
                                    "Wasserkocher nehmen":0.0,
                                    "Wasserkocher befüllen und wieder abstellen":0.0,
                                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.0,
                                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.0,
                                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.0,
                                    "Wasser in die French Press geben und Wasserkocher abstellen":0.00,
                                    "3 minuten warten, dann French Press drücken":0.0,
                                    "French Press nehmen":0.0,
                                    "Kaffee servieren und French Press abstellen":0.1,
                                    "END":0.9
                                },
                                    "END":{
                                    "Wasserkocher nehmen":0.0,
                                    "Wasserkocher befüllen und wieder abstellen":0.0,
                                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.0,
                                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.0,
                                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.0,
                                    "Wasser in die French Press geben und Wasserkocher abstellen":0.00,
                                    "3 minuten warten, dann French Press drücken":0.0,
                                    "French Press nehmen":0.0,
                                    "Kaffee servieren und French Press abstellen":0.0,
                                    "END":1.0
                                }

                            }
       
        return transition_probabilities

    def set_default_emission_probabilities(self):

        emission_probabilities = {
                    "Wasserkocher nehmen":{
                        'boiler_full_water':0.0,
                        'boiler_in_place':0.0,
                        'boiler_in_place-boiler_full_water':0.0,
                        'french_press_in_place':0.0,
                        'french_press_in_place-boiler_full_water':0.0,
                        'french_press_in_place-boiler_in_place':0.0,
                        'french_press_in_place-boiler_in_place-boiler_full_water':0.0,
                        'coffee_in_place':0.0,
                        'coffee_in_place-boiler_full_water':0.0,
                        'coffee_in_place-boiler_in_place':0.0,
                        'coffee_in_place-boiler_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place':1.0,
                        'coffee_in_place-french_press_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water':0.0
                    },

                    "Wasserkocher befüllen und wieder abstellen":{
                        'boiler_full_water':0.0,
                        'boiler_in_place':0.0,
                        'boiler_in_place-boiler_full_water':0.0,
                        'french_press_in_place':0.0,
                        'french_press_in_place-boiler_full_water':0.0,
                        'french_press_in_place-boiler_in_place':0.0,
                        'french_press_in_place-boiler_in_place-boiler_full_water':0.0,
                        'coffee_in_place':0.0,
                        'coffee_in_place-boiler_full_water':0.0,
                        'coffee_in_place-boiler_in_place':0.0,
                        'coffee_in_place-boiler_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place':0.0,
                        'coffee_in_place-french_press_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water':1.0
                    },

                    "Wasserkocher anmachen, Kaffebehälter nehmen":{
                        'boiler_full_water':0.15,
                        'boiler_in_place':0.15,
                        'boiler_in_place-boiler_full_water':0.0,
                        'french_press_in_place':0.0,
                        'french_press_in_place-boiler_full_water':0.0,
                        'french_press_in_place-boiler_in_place':0.0,
                        'french_press_in_place-boiler_in_place-boiler_full_water':0.7,
                        'coffee_in_place':0.0,
                        'coffee_in_place-boiler_full_water':0.0,
                        'coffee_in_place-boiler_in_place':0.0,
                        'coffee_in_place-boiler_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place':0.0,
                        'coffee_in_place-french_press_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water':0.0
                    },

                    "Kaffeepulver in die French Press geben und Behälter abstellen":{
                        'boiler_full_water':0.0,
                        'boiler_in_place':0.0,
                        'boiler_in_place-boiler_full_water':0.0,
                        'french_press_in_place':0.0,
                        'french_press_in_place-boiler_full_water':0.0,
                        'french_press_in_place-boiler_in_place':0.0,
                        'french_press_in_place-boiler_in_place-boiler_full_water':0.0,
                        'coffee_in_place':0.0,
                        'coffee_in_place-boiler_full_water':0.0,
                        'coffee_in_place-boiler_in_place':0.0,
                        'coffee_in_place-boiler_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place':0.0,
                        'coffee_in_place-french_press_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water':1.0
                    },

                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":{
                        'boiler_full_water':0.1,
                        'boiler_in_place':0.0,
                        'boiler_in_place-boiler_full_water':0.0,
                        'french_press_in_place':0.1,
                        'french_press_in_place-boiler_full_water':0.0,
                        'french_press_in_place-boiler_in_place':0.0,
                        'french_press_in_place-boiler_in_place-boiler_full_water':0.0,
                        'coffee_in_place':0.0,
                        'coffee_in_place-boiler_full_water':0.0,
                        'coffee_in_place-boiler_in_place':0.0,
                        'coffee_in_place-boiler_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place':0.8,
                        'coffee_in_place-french_press_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water':1.0
                    },


                    "Wasser in die French Press geben und Wasserkocher abstellen":{
                            'boiler_full_water':0.0,
                            'boiler_in_place':0.0,
                            'boiler_in_place-boiler_full_water':0.0,
                            'french_press_in_place':0.0,
                            'french_press_in_place-boiler_full_water':0.0,
                            'french_press_in_place-boiler_in_place':0.0,
                            'french_press_in_place-boiler_in_place-boiler_full_water':0.0,
                            'coffee_in_place':0.0,
                            'coffee_in_place-boiler_full_water':0.0,
                            'coffee_in_place-boiler_in_place':0.0,
                            'coffee_in_place-boiler_in_place-boiler_full_water':0.0,
                            'coffee_in_place-french_press_in_place':0.0,
                            'coffee_in_place-french_press_in_place-boiler_full_water':0.0,
                            'coffee_in_place-french_press_in_place-boiler_in_place':0.5,
                            'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water':0.5
                        },

                    "3 minuten warten, dann French Press drücken":{
                            'boiler_full_water':0.0,
                            'boiler_in_place':0.0,
                            'boiler_in_place-boiler_full_water':0.0,
                            'french_press_in_place':0.0,
                            'french_press_in_place-boiler_full_water':0.0,
                            'french_press_in_place-boiler_in_place':0.0,
                            'french_press_in_place-boiler_in_place-boiler_full_water':0.0,
                            'coffee_in_place':0.0,
                            'coffee_in_place-boiler_full_water':0.0,
                            'coffee_in_place-boiler_in_place':0.0,
                            'coffee_in_place-boiler_in_place-boiler_full_water':0.0,
                            'coffee_in_place-french_press_in_place':0.0,
                            'coffee_in_place-french_press_in_place-boiler_full_water':0.0,
                            'coffee_in_place-french_press_in_place-boiler_in_place':0.8,
                            'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water':0.2
                        },

                    "French Press nehmen":{
                            'boiler_full_water':0.0,
                            'boiler_in_place':0.0,
                            'boiler_in_place-boiler_full_water':0.0,
                            'french_press_in_place':0.0,
                            'french_press_in_place-boiler_full_water':0.0,
                            'french_press_in_place-boiler_in_place':0.0,
                            'french_press_in_place-boiler_in_place-boiler_full_water':0.0,
                            'coffee_in_place':0.0,
                            'coffee_in_place-boiler_full_water':0.0,
                            'coffee_in_place-boiler_in_place':0.0,
                            'coffee_in_place-boiler_in_place-boiler_full_water':0.0,
                            'coffee_in_place-french_press_in_place':0.0,
                            'coffee_in_place-french_press_in_place-boiler_full_water':0.5,
                            'coffee_in_place-french_press_in_place-boiler_in_place':0.5,
                            'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water':0.0
                        },

                    "Kaffee servieren und French Press abstellen":{
                            'boiler_full_water':0.0,
                            'boiler_in_place':0.0,
                            'boiler_in_place-boiler_full_water':0.0,
                            'french_press_in_place':0.0,
                            'french_press_in_place-boiler_full_water':0.0,
                            'french_press_in_place-boiler_in_place':0.0,
                            'french_press_in_place-boiler_in_place-boiler_full_water':0.0,
                            'coffee_in_place':0.0,
                            'coffee_in_place-boiler_full_water':0.0,
                            'coffee_in_place-boiler_in_place':0.5,
                            'coffee_in_place-boiler_in_place-boiler_full_water':0.5,
                            'coffee_in_place-french_press_in_place':0.0,
                            'coffee_in_place-french_press_in_place-boiler_full_water':0.0,
                            'coffee_in_place-french_press_in_place-boiler_in_place':0.0,
                            'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water':1.0
                        },

                    "END":{
                        'boiler_full_water':0.0,
                        'boiler_in_place':0.1,
                        'boiler_in_place-boiler_full_water':0.3,
                        'french_press_in_place':0.3,
                        'french_press_in_place-boiler_full_water':0.0,
                        'french_press_in_place-boiler_in_place':0.0,
                        'french_press_in_place-boiler_in_place-boiler_full_water':0.0,
                        'coffee_in_place':0.0,
                        'coffee_in_place-boiler_full_water':0.0,
                        'coffee_in_place-boiler_in_place':0.3,
                        'coffee_in_place-boiler_in_place-boiler_full_water':0.3,
                        'coffee_in_place-french_press_in_place':0.0,
                        'coffee_in_place-french_press_in_place-boiler_full_water':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place':0.0,
                        'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water':0.0
                    }



            }
        
        return emission_probabilities

    def set_default_possible_sensor_observations(self):
        possible_sensor_observations = [
                                            'boiler_full_water',
                                            'boiler_in_place',
                                            'boiler_in_place-boiler_full_water',
                                            'french_press_in_place',
                                            'french_press_in_place-boiler_full_water',
                                            'french_press_in_place-boiler_in_place',
                                            'french_press_in_place-boiler_in_place-boiler_full_water',
                                            'coffee_in_place',
                                            'coffee_in_place-boiler_full_water',
                                            'coffee_in_place-boiler_in_place',
                                            'coffee_in_place-boiler_in_place-boiler_full_water',
                                            'coffee_in_place-french_press_in_place',
                                            'coffee_in_place-french_press_in_place-boiler_full_water',
                                            'coffee_in_place-french_press_in_place-boiler_in_place',
                                            'coffee_in_place-french_press_in_place-boiler_in_place-boiler_full_water'
                                            ]
        
        return possible_sensor_observations

    def set_default_start_probabilities(self):
    
        start_probabilities = {
                    "Wasserkocher nehmen":0.8,
                    "Wasserkocher befüllen und wieder abstellen":0.05,
                    "Wasserkocher anmachen, Kaffebehälter nehmen": 0.05,
                    "Kaffeepulver in die French Press geben und Behälter abstellen": 0.05,
                    "Nach dem Kochen des Wassers, Wasserkocher nehmen":0.00,
                    "Wasser in die French Press geben und Wasserkocher abstellen":0.05,
                    "3 minuten warten, dann French Press drücken":0.05,
                    "French Press nehmen":0.0,
                    "Kaffee servieren und French Press abstellen":0.0,
                    "END":0.0
                    }
        
        return start_probabilities
    
    def map_obs_vector_to_observationstr(self, header = None, vector = None):


        if not header:
           header = ['coffee_in_place', "french_press_in_place", "boiler_in_place", "boiler_full_water"]

        observation_elements = []

        for i, dimension in enumerate(vector):
            if dimension == "1":
                observation_elements.append(header[i])

        observation_string = "-".join(observation_elements)

        return observation_string

    def forward_algorithm_single_step(self, observation):
        """
        Forward Algorithm to compute the most probable state for a single observation.

        Parameters:
        - observation: The observed event at the current timestep
        - states: List of hidden states
        - start_prob: Initial state probabilities (dict)
        - trans_prob: Transition probabilities (dict of dicts)
        - emit_prob: Emission probabilities (dict of dicts)
        - prev_alpha: Previous alpha values for the recursion step (None if first observation)

        Returns:
        - The most probable state and its probability distribution
        """


        error_message = None

        states = self.states
        start_prob = self.start_probabilities
        trans_prob = self.transition_probabilities
        emit_prob = self.emission_probabilities
        prev_alpha= self.alpha_matrix


        n_states = len(states)
        
        # Initialize the alpha matrix for the new observation
        alpha = np.zeros(n_states)

        # Initialization step for the first observation
        if prev_alpha is None:
            for j, state in enumerate(states):
                alpha[j] = start_prob[state] * emit_prob[state][observation]
        else:
            # Recursion step for subsequent observations
            for j, state in enumerate(states):
                alpha[j] = sum(prev_alpha[i] * trans_prob[states[i]][state] for i in range(n_states)) * emit_prob[state][observation]

        # Termination step: Normalize the probabilities
        total_prob = sum(alpha)
        normalized_alpha = alpha / total_prob if total_prob != 0 else np.zeros(len(alpha))
        most_probable_state_probability_idx = np.argmax(normalized_alpha)
        
        # Get the most probable state
        most_probable_state =states[most_probable_state_probability_idx]
        most_probable_state_probability = normalized_alpha[most_probable_state_probability_idx]

        self.alpha_matrix = normalized_alpha

        if most_probable_state_probability <= 0.05:
           error_message = ERR_IMPOSSIBLE_STATE + self.last_state_prediction
        
        return most_probable_state, most_probable_state_probability, error_message # normalized_alpha
    

    def make_prediction_on_observation(self, observation_as_vector):

        error_messages = []

        observation_mapped = self.map_obs_vector_to_observationstr(header=None, vector = observation_as_vector)

        if observation_mapped not in self.possible_sensor_observations:
           error_messages.append(ERR_IMPOSSIBLE_OBS)
           observation_mapped = ERR_STATE
           most_probable_state = ERR_STATE
           most_probable_state_probability = ERR_STATE

        else:

            most_probable_state, most_probable_state_probability, error_message = self.forward_algorithm_single_step(observation_mapped)
            if error_message:
               error_messages.append(error_message)

        output_string = self.print_prediction(observation_mapped, most_probable_state, most_probable_state_probability, error_messages)
        #we want to recover later, so we don't save the current state
        if error_messages:
            self.last_state_prediction = most_probable_state
            self.last_state_prediction_probability = most_probable_state_probability

        return output_string

    def print_prediction(self, obs, most_probable_state, most_probable_state_probab, error_messages):

        if error_messages:
           str_err_message = ERR_MESSAGE_BEGIN + "\n"+"\n".join(error_messages)
           print(str_err_message)

        # print(f"{MOST_PROBABLE_STATE_MESSAGE} '{obs}': {most_probable_state}, ({PROBABILITY_MESSAGE}: {most_probable_state_probab})")
        output_string = f"{MOST_PROBABLE_STATE_MESSAGE} '{obs}': {most_probable_state}, ({PROBABILITY_MESSAGE}: {most_probable_state_probab})"
        print("Teo's code:" + output_string)
        return output_string


    
    def load_observation_sequence(self, filepath):
    
   
        obs_vectors_str = []
        # Open the CSV file
        with open(filepath, mode='r') as file:
            for line in file:
                line = line.strip()
                observation_vector = line.split(";")
                obs_vectors_str.append(observation_vector)


        return obs_vectors_str



model = PredictionModel()

# observations = model.load_observation_sequence("data3_error.csv")

# for o in observations:
#     model.make_prediction_on_observation(o)

