from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Defining the structure of the Bayesian Network
wildlife_model = BayesianNetwork([
    ("HabitatDestruction", "FoodAvailability"),
    ("ClimateChange", "FoodAvailability"),
    ("Pollution", "FoodAvailability"),
    ("Pollution", "PopulationHealth"),
    ("FoodAvailability", "PopulationHealth"),
    ("PopulationHealth", "PopulationSize")
])

# Defining CPDs
cpd_habitat_destruction = TabularCPD(variable="HabitatDestruction", variable_card=2, values=[[0.4], [0.6]])
cpd_climate_change = TabularCPD(variable="ClimateChange", variable_card=2, values=[[0.3], [0.7]])
cpd_pollution = TabularCPD(variable="Pollution", variable_card=2, values=[[0.2], [0.8]])

cpd_food_availability = TabularCPD(variable="FoodAvailability", variable_card=2,
                                   values=[[0.05, 0.4, 0.3, 0.4, 0.75, 0.85, 0.8, 0.9],
                                           [0.95, 0.6, 0.7, 0.6, 0.25, 0.15, 0.2, 0.1]],
                                   evidence=["HabitatDestruction", "ClimateChange", "Pollution"],
                                   evidence_card=[2, 2, 2])

cpd_population_health = TabularCPD(variable="PopulationHealth", variable_card=2,
                                   values=[[0.95, 0.1, 0.9, 0.2],
                                           [0.05, 0.9, 0.1, 0.8]],
                                   evidence=["FoodAvailability", "Pollution"],
                                   evidence_card=[2, 2])

cpd_population_size = TabularCPD(variable="PopulationSize", variable_card=2,
                                 values=[[0.7, 0.3],
                                         [0.3, 0.7]],
                                 evidence=["PopulationHealth"],
                                 evidence_card=[2])

# Associating the CPDs with the network
wildlife_model.add_cpds(cpd_habitat_destruction, cpd_climate_change, cpd_pollution,
                        cpd_food_availability, cpd_population_health, cpd_population_size)

# Checking if the model is valid
print("Model check:", wildlife_model.check_model())

# Viewing nodes of the model
print(wildlife_model.nodes())

# Viewing edges of the model
print(wildlife_model.edges())

# Listing all Independencies
print(wildlife_model.get_independencies())

# Performing inference
wildlife_infer = VariableElimination(wildlife_model)

# Example query: Probability of Population Size decrease given high pollution and climate change
#query_result = wildlife_infer.query(variables=["PopulationSize"],
                                     #evidence={"Pollution": 1, "ClimateChange": 1})
query_result = wildlife_infer.query(variables=["HabitatDestruction"])
#q = alarm_infer.query(variables=["Alarm", "Burglary"], evidence={"MaryCalls":0, "JohnCalls":0}    )
print(query_result)
