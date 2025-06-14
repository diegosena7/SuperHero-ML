import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Garante que matplotlib funcione sem interface gráfica
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer

# Leitura dos dados
DATA_PATH = './dados/superhero_abilities_dataset.csv'
df = pd.read_csv(DATA_PATH)

# Seleção de colunas relevantes
target = 'Alignment'
df = df.dropna(subset=['Strength', 'Speed', 'Intelligence', 'Combat Skill', 'Power Score', 'Popularity Score', 'Universe', 'Weapon', 'Alignment'])

# Análise exploratória e visualizações
exploratory_dir = os.path.abspath("./resultados/exploratoria/")
os.makedirs(exploratory_dir, exist_ok=True)

# Estatísticas descritivas
#print("Estatísticas descritivas:\n", df.describe(include='all'))

# Distribuição do target
plt.figure(figsize=(6,4))
df[target].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribuição das Classes (Alignment)')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.tight_layout()
plt.savefig(os.path.join(exploratory_dir, "distribuicao_classes.png"))
plt.close()

# Histogramas das features numéricas
df[['Strength', 'Speed', 'Intelligence', 'Combat Skill', 'Power Score', 'Popularity Score']].hist(
    figsize=(12,8), bins=15, color='orange')
plt.tight_layout()
plt.savefig(os.path.join(exploratory_dir, "histogramas_features.png"))
plt.close()
print("Visualizações e análise exploratória salvas em:", exploratory_dir)

# Encode categóricas
df_encoded = pd.get_dummies(df, columns=['Universe', 'Weapon'], drop_first=True)
print("Colunas após o encode:\n", df_encoded.head())

# Encode do target
le = LabelEncoder()
df_encoded[target] = le.fit_transform(df_encoded[target])

X = df_encoded.drop(columns=[target])
X = X.select_dtypes(include=[np.number])
y = df_encoded[target]

# Treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalonamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo com aprendizado de máquina
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Avaliação
y_pred = model.predict(X_test_scaled)
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Cria a pasta de resultados se não existir
output_dir = os.path.abspath("./resultados/")
os.makedirs(output_dir, exist_ok=True)

# Matriz de confusão
try:
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Matriz de confusão salva em: {conf_matrix_path}")
except Exception as e:
    print("Erro ao gerar matriz de confusão:", e)

# SHAP
try:
    print("Gerando gráfico SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    shap.summary_plot(shap_values, X_test, show=False)
    shap_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico SHAP salvo em: {shap_path}")
except Exception as e:
    print("Erro ao gerar gráfico SHAP:", e)

# LIME
try:
    print("Gerando explicação LIME...")
    explainer_lime = LimeTabularExplainer(
        X_train_scaled, 
        feature_names=X.columns.tolist(), 
        class_names=le.classes_, 
        discretize_continuous=True
    )
    exp = explainer_lime.explain_instance(X_test_scaled[0], model.predict_proba)
    lime_path = os.path.join(output_dir, "lime_explanation.html")
    exp.save_to_file(lime_path)
    print(f"Explicação LIME salva em: {lime_path}")
except Exception as e:
    print("Erro ao gerar explicação LIME:", e)

print("\nFinalizado com sucesso. Verifique os arquivos na pasta 'resultados'.")
# Fim do script