# --- IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS ---
import pandas as pd
import altair as alt
from collections import Counter
import re
from itertools import combinations
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import kagglehub
import os
import seaborn as sns
import geopandas as gpd

# --- 1. DOWNLOAD E CARREGAMENTO DOS DADOS ---
print("Baixando o dataset do Kaggle...")
path = kagglehub.dataset_download("ramjasmaurya/mineral-ores-around-the-world")
csv_file_path = os.path.join(path, "Mineral ores round the world.csv")
print(f"Path para o arquivo do dataset: {csv_file_path}")

try:
    df = pd.read_csv(csv_file_path, dtype={'prod_size': str, 'ore_ctrl': str})
    print("Dataset carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: O arquivo CSV não foi encontrado no path: {csv_file_path}")
    exit()

# --- 2. FILTRAR DADOS PARA O BRASIL ---
brazil_df = df[df['country'] == 'Brazil'].copy()
print(f"\nNúmero de sítios minerais no Brasil: {len(brazil_df)}")
brazil_df.dropna(subset=['latitude', 'longitude'], inplace=True)
print(f"Número de sítios no Brasil após a limpeza de dados de localização: {len(brazil_df)}")

# --- 3. ANÁLISE E VISUALIZAÇÃO GEOSESPACIAL ---
if not brazil_df.empty:
    print("\nGerando mapa dos sítios minerais do Brasil...")
    countries = alt.topo_feature('https://raw.githubusercontent.com/vega/vega-datasets/master/data/world-110m.json', 'countries')
    brazil_base = alt.Chart(countries).mark_geoshape(
        stroke='black', strokeWidth=1
    ).encode(
        color=alt.value('#E6E6E6')
    ).transform_filter(
        (alt.datum.id == 76)
    ).project(type='mercator')
    mineral_points = alt.Chart(brazil_df).mark_point(
        size=30, opacity=0.7, filled=True
    ).encode(
        longitude='longitude:Q', latitude='latitude:Q', color=alt.value('red'), tooltip=['site_name', 'commod1']
    )
    brazil_map = (brazil_base + mineral_points).properties(
        title='Ocorrências de Sítios Minerais no Brasil'
    )
    map_json_filename = "brazil_mineral_sites_map.json"
    brazil_map.save(map_json_filename)
    print(f"Mapa geoespacial interativo salvo como '{map_json_filename}'.")
    try:
        map_png_filename = "brazil_mineral_sites_map.png"
        brazil_map.save(map_png_filename)
        print(f"Mapa geoespacial estático salvo como '{map_png_filename}'.")
        plt.show()
    except Exception as e:
        print(f"Não foi possível salvar o mapa em PNG. Erro: {e}")
        print("Certifique-se de ter 'altair_saver' e 'vl-convert-python' instalados.")
else:
    print("\nNenhum sítio mineral encontrado no Brasil para mapear.")

# --- 4. ANÁLISE DE DADOS EXPLORATÓRIA (EDA) ---
print("\nIniciando Análise Exploratória de Dados (EDA) para o Brasil...")
mineral_counts = Counter()
mineral_cols = ['commod1', 'commod2', 'commod3']
for col in mineral_cols:
    brazil_df[col] = brazil_df[col].fillna('')
    for minerals_list in brazil_df[col].apply(lambda x: re.split(r',\s*', x.strip()) if x else []):
        for mineral in minerals_list:
            if mineral:
                mineral_counts[mineral.strip()] += 1
mineral_frequency_df = pd.DataFrame(mineral_counts.items(), columns=['Mineral', 'Count']).sort_values(by='Count', ascending=False)
top_10_minerals_brazil = mineral_frequency_df.head(10)['Mineral'].tolist()
print("\nTop 10 Minerais Mais Frequentes no Brasil:")
print(mineral_frequency_df.head(10))

print("\nAnalisando contexto geológico...")
rock_cols = ['arock_type', 'hrock_type']
co_occurrence_data = []
for _, row in brazil_df.iterrows():
    minerals_present = [m.strip() for col in mineral_cols for m in re.split(r',\s*', str(row[col])) if m.strip() in top_10_minerals_brazil and m.strip()]
    rocks_present = [r.strip() for col in rock_cols for r in re.split(r',\s*', str(row[col])) if r.strip() and str(row[col]) != 'nan']
    for mineral in set(minerals_present):
        for rock in set(rocks_present):
            co_occurrence_data.append({'Mineral': mineral, 'Rock_Type': rock})
if co_occurrence_data:
    co_occurrence_df = pd.DataFrame(co_occurrence_data)
    co_occurrence_counts = co_occurrence_df.groupby(['Mineral', 'Rock_Type']).size().reset_index(name='Count')
    top_10_rock_types = co_occurrence_counts.groupby('Rock_Type')['Count'].sum().nlargest(10).index.tolist()
    filtered_co_occurrence = co_occurrence_counts[co_occurrence_counts['Rock_Type'].isin(top_10_rock_types)]
    heatmap_geology = alt.Chart(filtered_co_occurrence).mark_rect().encode(
        x=alt.X('Rock_Type:N', title='Tipo de Rocha', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Mineral:N', title='Mineral'),
        color=alt.Color('Count:Q', title='Número de Ocorrências'),
        tooltip=['Mineral', 'Rock_Type', 'Count']
    ).properties(title='Co-ocorrência dos Top 10 Minerais e Rochas no Brasil')
    heatmap_filename_json = "brazil_geological_context_heatmap.json"
    heatmap_geology.save(heatmap_filename_json)
    print(f"Heatmap de contexto geológico salvo como '{heatmap_filename_json}'.")
    try:
        heatmap_filename_png = "brazil_geological_context_heatmap.png"
        heatmap_geology.save(heatmap_filename_png)
        print(f"Heatmap de contexto geológico estático salvo como '{heatmap_filename_png}'.")
        plt.show()
    except Exception as e:
        print(f"Não foi possível salvar o heatmap em PNG. Erro: {e}")
else:
    print("Nenhum dado de co-ocorrência mineral-rocha encontrado no Brasil.")

print("\nAnalisando co-ocorrência de minerais...")
co_occurrence_matrix = {mineral: {m: 0 for m in top_10_minerals_brazil} for mineral in top_10_minerals_brazil}
for _, row in brazil_df.iterrows():
    present_minerals = set()
    for col in mineral_cols:
        if pd.notna(row[col]) and row[col]:
            current_minerals = [m.strip() for m in re.split(r',\s*', str(row[col])) if m.strip() in top_10_minerals_brazil]
            present_minerals.update(current_minerals)
    for m1, m2 in combinations(sorted(list(present_minerals)), 2):
        co_occurrence_matrix[m1][m2] += 1
        co_occurrence_matrix[m2][m1] += 1
co_occurrence_df = pd.DataFrame(co_occurrence_matrix).reset_index().rename(columns={'index': 'Mineral1'})
co_occurrence_df_melted = co_occurrence_df.melt(id_vars='Mineral1', var_name='Mineral2', value_name='Count')
heatmap_composition = alt.Chart(co_occurrence_df_melted).mark_rect().encode(
    x=alt.X('Mineral1:N', title='Mineral 1', axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('Mineral2:N', title='Mineral 2'),
    color=alt.Color('Count:Q', title='Co-ocorrência'),
    tooltip=['Mineral1', 'Mineral2', 'Count']
).properties(title='Co-ocorrência de Top 10 Minerais no Brasil')
heatmap_composition_filename_json = "brazil_mineral_co_occurrence_heatmap.json"
heatmap_composition.save(heatmap_composition_filename_json)
print(f"Heatmap de co-ocorrência mineral salvo como '{heatmap_composition_filename_json}'.")
try:
    heatmap_composition_filename_png = "brazil_mineral_co_occurrence_heatmap.png"
    heatmap_composition.save(heatmap_composition_filename_png)
    print(f"Heatmap de co-ocorrência mineral estático salvo como '{heatmap_composition_filename_png}'.")
    plt.show()
except Exception as e:
    print(f"Não foi possível salvar o heatmap em PNG. Erro: {e}")

# --- 5. APLICAÇÕES DE MACHINE LEARNING (FOCO EM MINÉRIOS DE FERRO) ---
print("\nIniciando Aplicações de Machine Learning para minérios de Ferro (Fe)...")
iron_df = brazil_df.copy()
iron_df['is_iron_ore'] = iron_df.apply(
    lambda row: 1 if any('Iron' in str(row[col]) for col in mineral_cols) else 0, axis=1
)
print(f"Número de sítios com minério de Ferro: {iron_df['is_iron_ore'].sum()}")

clustering_features = ['latitude', 'longitude', 'arock_type', 'hrock_type']
iron_only_df = iron_df[iron_df['is_iron_ore'] == 1].dropna(subset=clustering_features)
categorical_features_cluster = ['arock_type', 'hrock_type']
preprocessor_clustering = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_cluster)
    ],
    remainder='passthrough'
)
X_cluster = preprocessor_clustering.fit_transform(iron_only_df[clustering_features])
if X_cluster.shape[0] > 0:
    # --- MÉTODOS PARA DEFINIR O 'K' ---
    print("\n--- Métodos para determinar o número ideal de clusters ('k') ---")

    inertia = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_cluster)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bx-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo para Agrupamento de Minérios de Ferro')
    elbow_filename = "elbow_method_plot.png"
    plt.savefig(elbow_filename)
    plt.show()
    plt.close()

    silhouette_scores = []
    k_range_sil = range(2, 10)
    for k in k_range_sil:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_cluster)
        score = silhouette_score(X_cluster, kmeans.labels_)
        silhouette_scores.append(score)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range_sil, silhouette_scores, 'bx-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Coeficiente de Silhueta')
    plt.title('Método da Silhueta para Agrupamento de Minérios de Ferro')
    silhouette_filename = "silhouette_method_plot.png"
    plt.savefig(silhouette_filename)
    plt.show()
    plt.close()

    # --- APLICAÇÃO DO K-MEANS E INTERPRETAÇÃO DOS CLUSTERS ---
    print("\nO valor ideal de 'k' para este dataset é 3.")
    k_ideal = 3
    print(f"\nAplicando K-Means com k = {k_ideal}...")
    kmeans_final = KMeans(n_clusters=k_ideal, random_state=42, n_init='auto')
    kmeans_final.fit(X_cluster)
    iron_only_df['cluster'] = kmeans_final.labels_

    cluster_descriptions = {}
    print("\nAnálise dos Clusters de Minério de Ferro:")
    for cluster_id in sorted(iron_only_df['cluster'].unique()):
        cluster_data = iron_only_df[iron_only_df['cluster'] == cluster_id]

        all_minerals = [m.strip() for col in ['commod1', 'commod2', 'commod3'] for m in cluster_data[col].astype(str).str.split(',').explode() if m.strip() and m.strip() not in ['Iron', 'nan']]
        if all_minerals:
            most_common_mineral = Counter(all_minerals).most_common(3)
        else:
            most_common_mineral = []

        all_rocks = [r.strip() for col in ['arock_type', 'hrock_type'] for r in cluster_data[col].astype(str).str.split(',').explode() if r.strip() not in ['nan', 'Unknown']]
        if all_rocks:
            most_common_rock = Counter(all_rocks).most_common(3)
        else:
            most_common_rock = []

        avg_lat = cluster_data['latitude'].mean()
        avg_lon = cluster_data['longitude'].mean()
        most_common_dev_stat = cluster_data['dev_stat'].mode().iloc[0] if not cluster_data['dev_stat'].mode().empty else 'N/A'

        description = f"""
Cluster {cluster_id}:
- Número de sítios: {len(cluster_data)}
- Rochas mais comuns: {', '.join([f'{rock[0]} ({rock[1]})' for rock in most_common_rock]) if most_common_rock else 'Nenhuma rocha associada encontrada.'}
- Minerais associados: {', '.join([f'{mineral[0]} ({mineral[1]})' for mineral in most_common_mineral]) if most_common_mineral else 'Nenhum mineral associado encontrado.'}
- Centro geográfico aproximado: Lat {avg_lat:.2f}, Lon {avg_lon:.2f}
- Status de desenvolvimento mais comum: {most_common_dev_stat}
"""
        cluster_descriptions[cluster_id] = description
        print(description)

    print("\nGerando mapa dos clusters e descrições...")
    brazil_geojson_url = 'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-mun.json'
    try:
        brazil_map_gdf = gpd.read_file(brazil_geojson_url)
    except Exception as e:
        print(f"Erro ao carregar o GeoJSON do Brasil para o Geopandas: {e}")
        print("Não será possível gerar o mapa com o contorno do país.")
        brazil_map_gdf = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    if brazil_map_gdf is not None:
        brazil_map_gdf.plot(ax=ax1, color='lightgray', edgecolor='black', alpha=0.3)
        ax1.set_title('Clusters de Sítios de Minério de Ferro no Brasil', fontsize=16)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(True, linestyle='--', alpha=0.6)

        cluster_colors = plt.cm.get_cmap('viridis', k_ideal)
        for cluster_id in sorted(iron_only_df['cluster'].unique()):
            cluster_data = iron_only_df[iron_only_df['cluster'] == cluster_id]
            ax1.scatter(cluster_data['longitude'], cluster_data['latitude'],
                        color=cluster_colors(cluster_id), s=50, alpha=0.8, label=f'Cluster {cluster_id}')

            if len(cluster_data) > 1:
                cov = np.cov(cluster_data[['longitude', 'latitude']].T)
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(eigenvals) * 2
                ell = Ellipse(xy=(cluster_data['longitude'].mean(), cluster_data['latitude'].mean()),
                              width=width, height=height, angle=angle,
                              edgecolor=cluster_colors(cluster_id), facecolor='none', linestyle='--', linewidth=2)
                ax1.add_patch(ell)

        ax1.legend(title="Clusters", loc='upper right')

    else:
        ax1.scatter(iron_only_df['longitude'], iron_only_df['latitude'], c=iron_only_df['cluster'],
                    cmap='viridis', s=50, alpha=0.8)
        ax1.set_title('Clusters de Sítios de Minério de Ferro no Brasil (Sem contorno do mapa)', fontsize=16)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(True, linestyle='--', alpha=0.6)

    text_content = "\n".join(cluster_descriptions.values())
    ax2.text(0.05, 0.95, text_content, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    ax2.set_title('Análise Descritiva dos Clusters', fontsize=16)
    ax2.axis('off')

    plt.tight_layout()
    cluster_viz_filename = "brazil_iron_clusters_and_analysis_combined.png"
    plt.savefig(cluster_viz_filename, bbox_inches='tight')
    print(f"Mapa dos clusters com descrições salvo como '{cluster_viz_filename}'.")
    plt.show()
    plt.close()

else:
    print("Não há dados de minério de Ferro suficientes para o agrupamento.")

# --- Análise de PCA para depósitos em desenvolvimento de Ferro (Fe) ---
print("\n--- Análise de PCA para depósitos em desenvolvimento de Ferro (Fe) ---")
pca_df = iron_df[(iron_df['is_iron_ore'] == 1) & (iron_df['dev_stat'] == 'Producer')].copy()
if not pca_df.empty:
    pca_features = ['latitude', 'longitude', 'arock_type', 'hrock_type']
    for col in ['arock_type', 'hrock_type']:
        pca_df[col] = pca_df[col].fillna('Unknown')
    X_pca_df = pca_df[pca_features]
    categorical_features_pca = ['arock_type', 'hrock_type']
    numerical_features_pca = ['latitude', 'longitude']
    preprocessor_pca = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_pca),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_pca)
        ])
    X_pca_preprocessed = preprocessor_pca.fit_transform(X_pca_df)
    pca = PCA(n_components=2)
    X_pca_transformed = pca.fit_transform(X_pca_preprocessed)
    explained_variance = pca.explained_variance_ratio_
    print(f"\nVariância explicada por cada componente principal: {explained_variance}")
    feature_names = preprocessor_pca.get_feature_names_out()
    pca_components_df = pd.DataFrame(pca.components_, columns=feature_names, index=['PC1', 'PC2'])

    pca_descriptions = {}

    # Análise para o PC1
    pc1_top_features = pca_components_df.loc['PC1'].sort_values(ascending=False).head(5)
    pc1_text = f"""
Componente Principal 1 (PC1) - {explained_variance[0] * 100:.2f}% da Variância:
Esta componente está fortemente ligada a fatores geográficos, em especial a Longitude. As rochas de Quartzito, Granito e Filito também são importantes, mas com um peso menor.
O PC1 representa uma variação na distribuição longitudinal dos depósitos e nas associações rochosas mais comuns.
Variáveis mais importantes:
{pc1_top_features.to_string()}
"""
    pca_descriptions['PC1'] = pc1_text

    # Análise para o PC2
    pc2_top_features = pca_components_df.loc['PC2'].sort_values(ascending=False).head(5)
    pc2_text = f"""
Componente Principal 2 (PC2) - {explained_variance[1] * 100:.2f}% da Variância:
A segunda componente principal também está relacionada à geografia, mas com a Latitude e Longitude tendo pesos muito próximos.
O PC2 é altamente influenciado por formações de Ferro (`Iron Formation`) e rochas de Quartzito e Granito, indicando que representa variações no tipo de rocha hospedeira e associada nos depósitos.
Variáveis mais importantes:
{pc2_top_features.to_string()}
"""
    pca_descriptions['PC2'] = pc2_text

    print("\nGerando gráfico da variância explicada pelo PCA com descrições...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Gráfico de barras da variância
    ax1.bar(range(len(explained_variance)), explained_variance, alpha=0.5, align='center',
            label='Variância Individual Explicada')
    ax1.set_ylabel('Variância Explicada')
    ax1.set_xlabel('Componente Principal')
    ax1.set_title('Variância Explicada por Componente Principal', fontsize=16)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Texto dissertativo sobre a PCA
    text_content = "\n".join(pca_descriptions.values())
    ax2.text(0.05, 0.95, text_content, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    ax2.set_title('Análise Descritiva do PCA', fontsize=16)
    ax2.axis('off')

    plt.tight_layout()
    pca_filename = "pca_explained_variance_with_text.png"
    plt.savefig(pca_filename, bbox_inches='tight')
    print(f"Análise de PCA salva como '{pca_filename}'.")
    plt.show()
    plt.close()
else:
    print("Não há dados para a Análise de PCA com os filtros selecionados.")

# --- 5.2. Modelagem Preditiva (produtividade e desenvolvimento) ---
print("\nPreparando para a Modelagem Preditiva (Previsão de 'dev_stat')...")
iron_df_modeling = iron_df.dropna(subset=['dev_stat'])
if iron_df_modeling.empty:
    print("Não há dados de 'dev_stat' suficientes para a modelagem preditiva.")
else:
    # --- Incluindo o cluster como feature ---
    # Adicionando o cluster ao DataFrame principal para ser usado no modelo
    iron_df_modeling = iron_df_modeling.merge(iron_only_df[['cluster']], left_index=True, right_index=True, how='left')
    iron_df_modeling['cluster'] = iron_df_modeling['cluster'].fillna(-1).astype(int)
    
    features = ['latitude', 'longitude', 'arock_type', 'hrock_type', 'cluster']
    target = 'dev_stat'
    X = iron_df_modeling[features]
    y = iron_df_modeling[target]
    
    categorical_features = ['arock_type', 'hrock_type', 'cluster']
    preprocessor_modeling = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Grid Search para otimização ---
    print("\nIniciando Grid Search para otimizar hiperparâmetros...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_modeling),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, None]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print("\nMelhores hiperparâmetros encontrados:")
    print(grid_search.best_params_)
    
    # --- Avaliação do melhor modelo ---
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModelo otimizado treinado para prever 'dev_stat'.")
    print(f"Acurácia do modelo otimizado: {accuracy:.2f}")
    
    print("\nGerando Matriz de Confusão para avaliação do modelo...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_pipeline.classes_, yticklabels=best_pipeline.classes_,
                ax=ax)
    plt.xlabel('Previsão do Modelo')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusão para Previsão de Status de Desenvolvimento (Otimizado)')
    cm_filename = "confusion_matrix_optimized.png"
    plt.savefig(cm_filename)
    print(f"Matriz de Confusão salva como '{cm_filename}'.")
    plt.show()
    plt.close()

    # --- Previsão e Visualização das Top 5 Localidades ---
    print("\n--- Previsão de Localidades Produtivas para Minério de Ferro ---")
    
    # Filtra apenas os sítios com minério de ferro que ainda não são "Producer"
    potential_sites = iron_df_modeling[iron_df_modeling['dev_stat'] != 'Producer'].copy()
    
    if not potential_sites.empty:
        # Prepara os dados para a previsão
        X_predict = potential_sites[features]
        
        # O modelo otimizado prevê a probabilidade para cada classe
        probabilities = best_pipeline.predict_proba(X_predict)
        
        # Encontra a coluna correspondente à classe 'Producer'
        producer_class_index = np.where(best_pipeline.classes_ == 'Producer')[0][0]
        
        # Extrai as probabilidades da classe 'Producer'
        producer_probabilities = probabilities[:, producer_class_index]
        
        # Adiciona as probabilidades ao DataFrame
        potential_sites['producer_probability'] = producer_probabilities
        
        # Seleciona as 5 localidades com maior probabilidade
        top_5_potential = potential_sites.nlargest(5, 'producer_probability')
        
        print("\nAs 5 localidades com maior probabilidade de serem depósitos de minério de ferro produtivos são:")
        top_5_potential_filename = "top_5_potential_iron_sites.txt"
        with open(top_5_potential_filename, "w") as f:
            for index, row in top_5_potential.iterrows():
                site_info = (
                    f"Site: {row['site_name']} "
                    f"(Lat: {row['latitude']:.2f}, Lon: {row['longitude']:.2f})\n"
                    f"Probabilidade de ser Produtor: {row['producer_probability'] * 100:.2f}%\n"
                    f"Tipo de Rocha Associada: {row['arock_type']}\n"
                    f"Tipo de Rocha Hospedeira: {row['hrock_type']}\n"
                    f"---"
                )
                print(site_info)
                f.write(site_info + "\n")
        print(f"\nDetalhes das top 5 localidades salvas em '{top_5_potential_filename}'.")

        # --- Visualização das Top 5 no Mapa ---
        print("\nGerando mapa com as 5 localidades mais promissoras...")
        brazil_geojson_url = 'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-mun.json'
        try:
            brazil_map_gdf = gpd.read_file(brazil_geojson_url)
        except Exception:
            brazil_map_gdf = None

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        if brazil_map_gdf is not None:
            brazil_map_gdf.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.3)
        else:
            ax.set_facecolor('lightgray')

        ax.set_title('Top 5 Localidades Mais Promissoras para Minério de Ferro Produtivo', fontsize=16)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        for idx, row in top_5_potential.iterrows():
            ax.scatter(row['longitude'], row['latitude'], color='red', s=200, alpha=0.8, edgecolor='black', linewidth=1)
            ax.annotate(f"{idx+1}", (row['longitude'], row['latitude']), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=12, fontweight='bold')
        
        # Adicionar rótulos aos pontos
        labels = [f"#{i+1}: {row['site_name']}\nProb: {row['producer_probability']:.2f}" for i, row in top_5_potential.iterrows()]
        ax.legend(handles=[plt.scatter([], [], color='red', s=100, label=label) for label in labels], title='Localidades', loc='upper right')

        plt.tight_layout()
        top_5_map_filename = "brazil_top_5_iron_sites_map.png"
        plt.savefig(top_5_map_filename)
        print(f"Mapa das 5 localidades mais promissoras salvo como '{top_5_map_filename}'.")
        plt.show()
        plt.close()

    else:
        print("Nenhum sítio mineral que não seja produtor foi encontrado para análise preditiva.")

print("\nTodas as análises e visualizações foram realizadas com sucesso.")
