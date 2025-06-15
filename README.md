# Projeto de Machine Learning: Super-Heróis

## Descrição
Este projeto utiliza dados de super-heróis para treinar um modelo de classificação que prediz o alinhamento (Hero, Villain, Anti-Hero) com base em atributos como força, inteligência, popularidade, etc.

## Como Executar com Docker

1. Certifique-se de ter o Docker instalado.
2. No terminal, navegue até a pasta onde está o projeto e execute: 


## Resultados
Os seguintes arquivos serão gerados na pasta `resultados/`:
- `confusion_matrix.png`
- `shap_summary.png`
- `lime_explanation.html`

## Contato
Autores: Leonardo Amorim, Diego e Carol

```bash
docker build -t superheroes-ml .
docker run superheroes-ml
docker-compose up --build