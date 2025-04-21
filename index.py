import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import matplotlib.pyplot as plt

# Ler o CSV
df = pd.read_csv("student-scores 2.csv")
extracted_data = df[['absence_days', 'weekly_self_study_hours', 'extracurricular_activities']].dropna()
# Converter True/False para 1/0
extracted_data['extracurricular_activities'] = extracted_data['extracurricular_activities'].astype(int)

# Definir variáveis fuzzy
study_time = ctrl.Antecedent(np.arange(0, 51, 1), 'study_time')
absences = ctrl.Antecedent(np.arange(0, 11, 1), 'absences')
extracurricular = ctrl.Antecedent(np.arange(0, 2.01, 0.01), 'extracurricular')
recommendation = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation')

# Funções de pertinência
study_time['low'] = fuzz.trimf(study_time.universe, [0, 0, 20])
study_time['medium'] = fuzz.trimf(study_time.universe, [15, 25, 35])
study_time['high'] = fuzz.trimf(study_time.universe, [30, 50, 50])

absences['low'] = fuzz.trimf(absences.universe, [0, 0, 3])
absences['medium'] = fuzz.trimf(absences.universe, [2, 4, 6])
absences['high'] = fuzz.trimf   (absences.universe, [5, 10, 10])

extracurricular['no'] = fuzz.trimf(extracurricular.universe, [0, 0, 0.1])
extracurricular['yes'] = fuzz.trimf(extracurricular.universe, [0.9, 1, 1])

recommendation['basic'] = fuzz.trimf(recommendation.universe, [0, 0, 50])
recommendation['intermediate'] = fuzz.trimf(recommendation.universe, [30, 50, 70])
recommendation['advanced'] = fuzz.trimf(recommendation.universe, [70, 100, 100])

# Visualizar funções de pertinência
study_time.view()
absences.view()
extracurricular.view()
recommendation.view()

# plt.show()

# Definir regras
rules = [
    ctrl.Rule(study_time['low'] & absences['high'] & extracurricular['no'], recommendation['advanced']),
    ctrl.Rule(study_time['low'] & absences['low'] & extracurricular['yes'], recommendation['intermediate']),
    ctrl.Rule(study_time['medium'] & absences['medium'] & extracurricular['no'], recommendation['intermediate']),
    ctrl.Rule(study_time['high'] & absences['low'] & extracurricular['yes'], recommendation['basic']),
    ctrl.Rule(study_time['high'] & absences['high'] & extracurricular['no'], recommendation['advanced']),
    ctrl.Rule(study_time['medium'] & absences['high'] & extracurricular['yes'], recommendation['advanced']),
    ctrl.Rule(study_time['low'] & absences['low'] & extracurricular['no'], recommendation['intermediate']),
    ctrl.Rule(study_time['high'] & absences['high'] & extracurricular['yes'], recommendation['intermediate']),
    ctrl.Rule(study_time['medium'] & absences['low'] & extracurricular['no'], recommendation['intermediate']),
    ctrl.Rule(study_time['medium'] & absences['low'] & extracurricular['yes'], recommendation['basic']),
    ctrl.Rule(study_time['low'] & absences['medium'] & extracurricular['no'], recommendation['advanced']),
    ctrl.Rule(study_time['low'] & absences['medium'] & extracurricular['yes'], recommendation['intermediate']),
    ctrl.Rule(study_time['medium'] & absences['medium'] & extracurricular['yes'], recommendation['intermediate']),
    ctrl.Rule(study_time['medium'] & absences['high'] & extracurricular['no'], recommendation['advanced']),
    ctrl.Rule(study_time['high'] & absences['medium'] & extracurricular['no'], recommendation['intermediate']),
    ctrl.Rule(study_time['high'] & absences['medium'] & extracurricular['yes'], recommendation['basic']),
    ctrl.Rule(study_time['low'] & absences['high'] & extracurricular['yes'], recommendation['intermediate']),
    ctrl.Rule(study_time['high'] & absences['low'] & extracurricular['no'], recommendation['basic']),
]

# Sistema fuzzy
recommendation_ctrl = ctrl.ControlSystem(rules)
simulator = ctrl.ControlSystemSimulation(recommendation_ctrl)

# Processar os dados
recommendations = []
for _, row in extracted_data.iterrows():
    try:
        simulator.input['study_time'] = row['weekly_self_study_hours']
        simulator.input['absences'] = row['absence_days']
        simulator.input['extracurricular'] = row['extracurricular_activities']
        simulator.compute()
        recommendations.append(math.ceil(simulator.output['recommendation'])) 
    except:
        recommendations.append(None)

# Salvar resultados
extracted_data['recommendation'] = recommendations
extracted_data.to_csv("fuzzy_recommendations_output.csv", index=False)
print("Arquivo salvo como fuzzy_recommendations_output.csv")

# Contar as categorias de recomendação
high_recommendation = len(extracted_data[extracted_data['recommendation'] >= 70])
medium_recommendation = len(extracted_data[(extracted_data['recommendation'] >= 30) & (extracted_data['recommendation'] < 70)])
low_recommendation = len(extracted_data[extracted_data['recommendation'] < 30])

# Calcular as porcentagens
total = len(extracted_data)
high_percentage = (high_recommendation / total) * 100
medium_percentage = (medium_recommendation / total) * 100
low_percentage = (low_recommendation / total) * 100

# Exibir os resultados
print(f"Porcentagem de alta recomendação de intervenção: {high_percentage:.2f}%")
print(f"Porcentagem de média recomendação de intervenção: {medium_percentage:.2f}%")
print(f"Porcentagem de baixa ou nenhuma recomendação de intervenção: {low_percentage:.2f}%")