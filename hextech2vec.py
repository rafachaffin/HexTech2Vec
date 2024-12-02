import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import re

class Hextech2Vec:
    def __init__(self, ability_weight=2.5, stats_weight=0.5, use_pca=False, n_components=2):
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.categorical_features = ['classes', 'posicoes', 'tipo_range', 'tipo_dano']
        self.ability_types = ['passiva', 'q', 'w', 'e', 'r']
        self.ability_weight = ability_weight
        self.stats_weight = stats_weight
        self.ability_vectors = {}
        self.frobenius = 0
        
        
        self.use_pca = use_pca
        self.ability_pca = PCA(n_components=n_components) if use_pca else None
        self.use_ability_pca = use_pca
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.n_components = n_components

        
        self.ability_keywords = {
            'damage': ['damage', 'strike', 'hit', 'blast', 'attack', 'slash', 'deal', 'inflict'],
            'cc': ['stun', 'slow', 'immobilize', 'root', 'silence', 'knockup', 'snare', 'suppress', 'airborne', 'charm', 'taunt', 'fear', 'polymorph'],
            'mobility': ['dash', 'jump', 'leap', 'teleport', 'blink', 'movement', 'speed', 'flash', 'charge'],
            'shield': ['shield', 'barrier', 'protection', 'defend', 'block'],
            'heal': ['heal', 'restore', 'regenerate', 'recovery', 'health'],
            'buff': ['increase', 'bonus', 'amplify', 'enhance', 'boost', 'empower'],
            'ultimate': ['ultimate', 'supreme', 'final', 'devastating'],
            'aoe': ['area', 'radius', 'nearby', 'surrounding', 'around'],
            'projectile': ['projectile', 'missile', 'bolt', 'shot', 'throw'],
            'passive': ['passive', 'innate', 'permanently', 'constantly'],
            'debuff': ['weaken', 'reduce', 'decrease', 'diminish', 'impair'],
            'stealth': ['stealth', 'invisible', 'camouflage', 'unseen', 'hide'],
            'execute': ['execute', 'finish', 'killing', 'eliminate', 'lethal'],
            'sustain': ['sustain', 'lifesteal', 'drain', 'absorb', 'leech']
        }

    def _preprocess_ability_text(self, text):
        if not isinstance(text, str):
            return ""

        
        text = text.lower()

        
        for category, keywords in self.ability_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    text += f" [TYPE_{category.upper()}]"

        
        text = re.sub(r'[^\w\s\[\]_]', ' ', text)

        return text

    def _normalize_vector(self, vector):
        """Normaliza um vetor dividindo pelo seu comprimento (norma)"""
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector

    def fit_transform(self, df):
        processed_df = df.copy()

        
        encoded_features = np.zeros((len(processed_df), len(self.categorical_features)))
        for i, feature in enumerate(self.categorical_features):
            le = LabelEncoder()
            encoded_features[:, i] = le.fit_transform(processed_df[feature])
            self.label_encoders[feature] = le

        
        all_abilities = []
        for ability in self.ability_types:
            processed_abilities = processed_df[f'habilidade_{ability}'].apply(self._preprocess_ability_text)
            all_abilities.extend(processed_abilities.tolist())

        
        self.tfidf_vectorizer.fit(all_abilities)

        
        champion_embeddings = {}
        feature_names = []
        for idx, row in processed_df.iterrows():
            champion = row['nome']
            
            ability_vectors = []
            for ability in self.ability_types:
                ability_text = row[f'habilidade_{ability}']
                processed_text = self._preprocess_ability_text(ability_text)
                vector = self.tfidf_vectorizer.transform([processed_text]).toarray()[0]

                
                if ability == 'r':
                    vector *= 1.5

                self.ability_vectors[champion] = self.ability_vectors.get(champion, {})
                self.ability_vectors[champion][ability] = vector
                ability_vectors.append(vector)
            
                

            
            ability_matrix = np.array(ability_vectors)
            mean_ability_vector = np.mean(ability_matrix, axis=0) * self.ability_weight

            
            categorical_vector = encoded_features[idx] * self.stats_weight

            
            final_vector = np.concatenate([
                categorical_vector.reshape(-1),
                mean_ability_vector.reshape(-1)
            ])

            
            if idx == 0:
                feature_names = (
                    self.categorical_features + 
                    [f'ability_feature_{i}' for i in range(len(mean_ability_vector))]
                )

            
            champion_embeddings[champion] = self._normalize_vector(final_vector)
        
        
        if self.use_pca:
            champion_names = list(champion_embeddings.keys())
            vectors_champion = np.array([champion_embeddings[champ] for champ in champion_names])
            
            reduced_vectors = self.pca.fit_transform(vectors_champion)

            
            champion_embeddings = {
                champ: vec for champ, vec in zip(champion_names, reduced_vectors)
            }

            all_ability_vectors = []
            for champ in self.ability_vectors:
                for ability in self.ability_types:
                    all_ability_vectors.append(self.ability_vectors[champ][ability])
            
            
            ability_pca_vectors = self.ability_pca.fit_transform(all_ability_vectors)
            
            
            pca_idx = 0
            for champ in self.ability_vectors:
                for ability in self.ability_types:
                    self.ability_vectors[champ][ability] = ability_pca_vectors[pca_idx]
                    pca_idx += 1
        
        return champion_embeddings, feature_names


    def analyze_components(self, feature_names):
        """
        Analisa a importância das features originais em cada componente
        """
        if not self.use_pca:
            raise ValueError("PCA não está habilitado. Use use_pca=True no construtor.")
        
        component_analysis = []
        
        for i in range(self.n_components):
            
            coefficients = self.pca.components_[i]
            
            
            feature_importance = list(zip(feature_names, coefficients))
            
            
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            component_analysis.append({
                'component': i+1,
                'explained_variance': self.pca.explained_variance_ratio_[i],
                'top_features': feature_importance[:10]  
            })
            
        return component_analysis

    def compare_abilities(self, champ1, ability1, champ2, ability2):
        """Compara duas habilidades específicas com métricas detalhadas"""
        vec1 = self.ability_vectors[champ1][ability1]
        vec2 = self.ability_vectors[champ2][ability2]

        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        
        common_types = []
        for category in self.ability_keywords.keys():
            has_type1 = any(keyword in self._preprocess_ability_text(champ1)
                           for keyword in self.ability_keywords[category])
            has_type2 = any(keyword in self._preprocess_ability_text(champ2)
                           for keyword in self.ability_keywords[category])
            if has_type1 and has_type2:
                common_types.append(category)

        return {
            'similarity': similarity,
            'common_types': common_types,
            'weighted_similarity': similarity * (1 + 0.1 * len(common_types))
        }

    def _normalize_vector(self, vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector

    def find_similar_abilities_weighted(self, champion, threshold=0.5, ability_importance={
        'passiva': 1.0,
        'q': 1.2,
        'w': 1.2,
        'e': 1.2,
        'r': 1.5
    }):
        similar_abilities = {}

        for ability_type in self.ability_types:
            target_vector = self.ability_vectors[champion][ability_type]
            similar_abilities[ability_type] = []

            for other_champ in self.ability_vectors:
                if other_champ != champion:
                    for other_ability in self.ability_types:
                        comparison = self.compare_abilities(champion, ability_type,
                                                         other_champ, other_ability)

                        
                        if ability_type != other_ability:
                            comparison['weighted_similarity'] *= 0.8  

                        if comparison['weighted_similarity'] > threshold:
                            similar_abilities[ability_type].append({
                                'champion': other_champ,
                                'ability': other_ability,
                                'raw_similarity': comparison['similarity'],
                                'weighted_similarity': comparison['weighted_similarity'],
                                'common_types': comparison['common_types']
                            })

            similar_abilities[ability_type].sort(key=lambda x: x['weighted_similarity'], reverse=True)

        return similar_abilities

    def get_similar_champions(self, champion_embeddings_tupla, champion_name, top_n=5):
        """Encontra campeões similares baseado nos embeddings"""
        champion_embeddings = champion_embeddings_tupla[0]
        
        
        if not isinstance(champion_embeddings, dict):
            
            champion_embeddings = dict(zip(champion_embeddings.keys(), champion_embeddings))

        target_vector = champion_embeddings[champion_name]

        similarities = {}
        for champ, vector in champion_embeddings.items():
            if champ != champion_name:
                similarity = np.dot(target_vector, vector)
                similarities[champ] = similarity

        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def visualize_ability_similarities(self, champ1, champ2):
        """Cria um heatmap das similaridades entre habilidades"""
        similarity_matrix = np.zeros((len(self.ability_types), len(self.ability_types)))

        for i, ability1 in enumerate(self.ability_types):
            for j, ability2 in enumerate(self.ability_types):
                comparison = self.compare_abilities(champ1, ability1, champ2, ability2)
                similarity_matrix[i, j] = comparison['weighted_similarity']

       
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        
        ax.tick_params(axis="x", colors="white")  
        ax.tick_params(axis="y", colors="white")
        
        
        sns.heatmap(
            similarity_matrix,
            annot=True,
            cmap=sns.color_palette("Blues", as_cmap=True),
            xticklabels=self.ability_types,
            yticklabels=self.ability_types,
            ax=ax,
            cbar_kws={"ticks": []}  
        )

        
        colorbar = ax.collections[0].colorbar  
        colorbar.ax.tick_params(colors="white")  
        

        
        plt.title(f'Similaridade de Habilidades: {champ1} vs {champ2}', color="white")
        plt.xlabel(f'Habilidades de {champ2}', color="white")
        plt.ylabel(f'Habilidades de {champ1}', color="white")
        
        
        return fig

    def get_frobenius(self, embeddings):
        dicionario = embeddings[0]
        matrix = [vetor for vetor in dicionario.values()]

        if(self.frobenius == 0):
            self.frobenius = np.linalg.norm(matrix)
            return self.frobenius
        else:
            return self.frobenius
        
    def get_frobenius_percentage(self , pca_norm, matrix_norm ):

        return (pca_norm*100)/matrix_norm

      