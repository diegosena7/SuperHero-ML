# Projeto de Machine Learning: Super-Heróis

## Descrição
Este projeto utiliza dados de super-heróis para treinar um modelo de classificação que prediz o alinhamento (Hero, Villain, Anti-Hero) com base em atributos como força, inteligência, popularidade, etc.

## Como Executar com Docker

1. Certifique-se de ter o Docker instalado.
2. No terminal, navegue até a pasta onde está o projeto e execute:

```bash
docker build -t superheroes-ml .
docker run superheroes-ml