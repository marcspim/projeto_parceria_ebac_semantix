# Projeto de Prospecção de Minério de Ferro no Brasil

Este projeto utiliza técnicas de Mineração de Dados e Aprendizado de Máquina para identificar padrões geológicos e prever novos sítios com potencial produtivo de minério de ferro (Fe) no Brasil. A análise combina EDA, geoprocessamento e modelos preditivos para apoiar decisões estratégicas no setor mineral.

---

## Objetivo

O projeto tem como objetivo identificar áreas promissoras para prospecção de minério de ferro utilizando:

- PCA (Análise de Componentes Principais)
- K-Means (Agrupamento)
- Random Forest (Classificação)

O foco está na transformação de dados geológicos brutos em insights estratégicos para prospecção mineral.

---

## Coleta e Preparação dos Dados

- Fonte: Dataset "Mineral ores round the world" (Kaggle)
- Filtro: Apenas sítios minerais localizados no Brasil
- Limpeza:
  - Remoção de registros sem coordenadas
  - Padronização de minerais e tipos de rochas
- Total analisado: 480 sítios com coordenadas válidas

---

## Análise Exploratória (EDA)

### Mapeamento Geoespacial
A maior densidade de sítios minerais encontra-se nas regiões Sul, Sudeste e partes do Norte e Nordeste.

### Minerais Mais Frequentes

| Mineral | Contagem |
|--------|----------|
| Gold | 277 |
| Mica | 185 |
| Diamond | 158 |
| Iron | 129 |
| Tin | 110 |
| Copper | 88 |

### Relações Geológicas
- Mica associada a Pegmatite e Schist
- Gold e Diamond associados a depósitos aluviais (Gravel e Alluvium)
- Iron fortemente associado a Iron Formation

---

## Modelagem

### 1. PCA e K-Means

O objetivo do agrupamento foi identificar padrões geológicos entre depósitos de minério de ferro.

- Número ótimo de clusters: k = 3
- PC1 (46,71%): Longitude + Quartzite, Granite, Phyllite
- PC2 (24,06%): Latitude/Longitude + Iron Formation, Quartzite, Granite

#### Clusters Identificados

| Cluster | N. Sítios | Rochas Comuns | Minerais Associados | Status |
|--------|-----------|----------------|----------------------|--------|
| 0 | 6 | Dunite, Ultramafic Intrusive Rock, Norite | Nickel, Cobalt | Ocorrência |
| 1 | 31 | Iron Formation, Quartzite, Granite | Gold, Nickel | Produtor |
| 2 | 2 | Arkose, Sandstone, Tuff | Manganese | Produtor |

O Cluster 1 representa a principal região produtiva de ferro. O Cluster 0 representa ocorrências ultramáficas raras.

---

### 2. Random Forest (Classificação)

- Objetivo: prever se um sítio pode alcançar o status de Produtor
- Modelo otimizado via Grid Search
- Acurácia: 74%
- Bom desempenho para as classes Producer e Occurrence

---

## Top 5 Localidades Mais Promissoras

| Rank | Localidade | Probabilidade | Rocha Hospedeira |
|------|------------|---------------|-------------------|
| 1 | Damasio | 100.00% | Metavolcanic Rock, Conglomerate |
| 2 | Corrego Velho | 99.50% | Metavolcanic Rock, Conglomerate |
| 3 | Mundo Velho | 97.50% | Gneiss, Pegmatite |
| 4 | Paqueiro Lead Deposit | 95.00% | Limestone |
| 5 | Capitao Do Mato | 92.00% | Iron Formation |

### Insight
As duas primeiras localidades (Damasio e Corrego Velho) devem ser priorizadas para investimento, indicando novos padrões em rochas metavulcânicas e conglomerados.

---

## Tecnologias Utilizadas

- Python
- Pandas, NumPy
- Scikit-Learn
- Geopandas, Folium
- Matplotlib

---

## Autor

[Marcel Sarcinelli Pimenta]
