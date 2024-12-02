import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from hextech2vec import Hextech2Vec

st.set_page_config(
    page_title="Hextech2Vec",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
st.markdown(
    """
    
    
    """,
    unsafe_allow_html=True
)


st.markdown("""
            <div >
                <h5 class="custom-title">Projeto Final</h5>
                <h1 class="custom-title">HexTech2Vec</h1>
            </div>
""", unsafe_allow_html=True)


st.markdown("""<p class ="intro">
Este site foi designado para o Projeto Final da matéria Computação Científica e Análise de Dados.<br>
            </p><p class="sub_intro">O trabalho consiste em transformar Campeões do <i>League of Legends</i> em vetores , usando métodos de processamento de linguagem natural, no caso o TF-IDF , assim permitindo assim uma comparação de classes , posições e habilidades atráves de similaridade de cossenos.</p>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    
    try:
        df = pd.read_csv('./champions_data.csv',sep = '|')  
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None


st.sidebar.markdown("""<h2>Ajustes</h2>""",unsafe_allow_html=True)



ability_weight = st.sidebar.slider(
    "Peso das Habilidades",
    min_value=0.1,
    max_value=5.0,
    value=2.5,
    step=0.1
)
stats_weight = st.sidebar.slider(
    "Peso dos Status",
    min_value=0.1,
    max_value=5.0,
    value=0.5,
    step=0.1
)

modelo = st.sidebar.selectbox(
    "Escolha o modelo",
    ("Hextech2Vec", "Hextech2Vec - PCA")
)



n = 2

df = load_data()

if df is not None:

    
    model = Hextech2Vec(ability_weight=ability_weight, stats_weight=stats_weight)
    
    
    with st.spinner("Gerando embeddings dos campeões..."):
        
        champion_embeddings = model.fit_transform(df)
        norm_frobenius = model.get_frobenius(champion_embeddings)

        if (modelo == "Hextech2Vec - PCA"):
            n = st.sidebar.number_input(
            "Escolha o número de PCA",
            value=2,

            )
            model = Hextech2Vec(ability_weight=3, stats_weight=0.1, use_pca = True ,n_components= n)
            champion_embeddings = model.fit_transform(df)
            norm_frobenius_pca = model.get_frobenius(champion_embeddings)

            st.sidebar.markdown("""
                                <h2 class="frobenius">A norma de Frobenius dessa matriz é %.2f </h2>
                                """%norm_frobenius_pca
                                ,unsafe_allow_html=True)
            
            st.sidebar.markdown("""
                                <h2 class="frobenius">Essa matriz está pegando %.2f%% do dado</h2>
                                """%model.get_frobenius_percentage(norm_frobenius_pca,norm_frobenius)
                                ,unsafe_allow_html=True)
            
            
        else:
            model = Hextech2Vec(ability_weight=3, stats_weight=0.1, use_pca = False )
            champion_embeddings = model.fit_transform(df)
            st.sidebar.markdown("""
                                <h2 class="frobenius">A norma de Frobenius dessa matriz é %.2f </h2>
                                """%model.get_frobenius(champion_embeddings)
                                ,unsafe_allow_html=True)
            

        

    
    
    tab1, tab2, tab3 = st.tabs([
        "Comparação de Campeões",
        "Análise de Habilidades",
        "Exemplos"
    ])
    
    
    with tab1:
        st.markdown("""<h2 class="func_title">Comparação de Campeões</h2>""",unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            champion1 = st.selectbox(
                "Selecione o primeiro campeão",
                options=sorted(df['nome'].unique()),
                key="champ1"
            )
        
        with col2:
            champion2 = st.selectbox(
                "Selecione o segundo campeão",
                options=sorted(df['nome'].unique()),
                key="champ2"
            )
        
        if st.button("Comparar Campeões"):
            
            similarities = model.get_similar_champions(
                champion_embeddings,
                champion1,
                top_n=3
            )
            
            col_text, col_viz = st.columns([1, 2])
        
            with col_text:
                st.markdown('<div style="text-align: center"><h3> Campeões similares a %s</h3> </div>'%champion1  ,True)
                for champ, similarity in similarities:
                    st.write(f"{champ}: {similarity:.3f}")
            
            with col_viz:
                st.markdown('<div style="text-align: center"><h3> Mapa de Calor de Similaridades</h3> </div>',True)
                try:
                    fig = model.visualize_ability_similarities(champion1, champion2)
                    
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Erro ao gerar visualização: {str(e)}")
    
    
    with tab2:
        st.markdown("""<h2 class="func_title">Análise de Habilidades</h2>""",unsafe_allow_html=True)
        champion = st.selectbox(
            "Selecione um campeão",
            options=sorted(df['nome'].unique()),
            key="champ_abilities"
        )
        
        threshold = st.slider(
            "Limiar de Similaridade",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        if st.button("Analisar Habilidades"):
            similar_abilities = model.find_similar_abilities_weighted(
                champion,
                threshold=threshold
            )
            
            for ability_type in model.ability_types:
                st.subheader(f"Habilidade {ability_type.upper()}")
                if similar_abilities[ability_type]:
                    for match in similar_abilities[ability_type][:3]:
                        with st.expander(
                            f"{match['ability'].upper()} de {match['champion']}"
                        ):
                            st.write(f"Similaridade: {match['weighted_similarity']:.3f}")
                            if match['common_types']:
                                st.write(f"Tipos em comum: {', '.join(match['common_types'])}")
                else:
                    st.write("Nenhuma habilidade similar encontrada.")
    
    
    with tab3:
        st.markdown("""<h2 class="func_title">Exemplo</h2>""",unsafe_allow_html=True)
        st.markdown("""
        <p>Para demonstrar como funciona a similaridade entres as habilidades irei mostrar a descrição da habilidade de um caso como exemplo , a matriz gerada pela comparação de habilidades
        , e um pequeno vídeo demonstrando a habilidade dos dois campeões.</p> 
        """,unsafe_allow_html=True)
        st.markdown(""" <h2 class = "sub_header_func_comp">AKSHAN X CAITLYN</h2>""",unsafe_allow_html=True)
        st.markdown("""<p>Caitlyn é uma atiradora que é atualmente jogado na rota inferior. Já o Akshan é também um atirador só que é jogado na rota do meio. Tanto as habilidades do Q , W , E e
                     Passiva não são parecidos entre si , porém a sua Habilidade Ultimate(R) são extremamente parecidas<p>""",unsafe_allow_html=True)
        
        col_3_1, col_3_2 = st.columns(2)

        with col_3_1:
            st.markdown(""" <h2 class = "sub_header_func">Akshan - Punição (Ultimate)</h2>""",unsafe_allow_html=True)
            with st.expander("Descrição Completa"):
                st.markdown("""
                **Descrição Oficial:**
                "Ativo: Akshan trava no campeão inimigo alvo e começa a Canalização  por 2.5 segundos,  revelando-o e revelando a si mesmo. Ele gradualmente armazena balas em sua arma ao longo da duração.
                Punição será reconjurado após a duração, ou pode ser reconjurado antes de 0.5 segundos. Punição é colocado em um tempo de recarga de  5 segundos se a canalização for cancelada.
                Reconjuração: Akshan dispara todas as balas armazenadas no alvo, cada uma concedendo brevemente visão ao redor de sua trajetória e causando dano físico ao primeiro inimigo atingido, aumentado em 0% − 300% (com base na vida perdida do alvo). Os tiros podem atingir estruturas.
                dano de cada bala aplica roubo de vida com 100% de eficácia e executa tropas."
                """)
                
            
        with col_3_2:
            st.markdown(""" <h2 class = "sub_header_func">Caitlyn - Ás na Manga (Ultimate)</h2>""",unsafe_allow_html=True)
            with st.expander("Descrição Completa"):
                st.markdown("""
                **Descrição Oficial:**
                "Ativo: Depois de um tempo, Caitlyn trava a mira em um campeão inimigo e carrega por 1 segundo. No inicio da conjuração, Caitlyn ganha visão verdadeira do alvo.
                Se Caitlyn termina de carregar, ela atira um projetil na direção do inimigo que causa dano físico ao primeiro campeão que acertar. Outros campeões inimigos podem interceptar o projetil."
                """)
            

        st.markdown("---")

        st.markdown(""" <h2 class = "sub_header_func_comp">MATRIZ DE HABILIDADES</h2>""",unsafe_allow_html=True)
        col_3_3, col_3_4 = st.columns(2)
        with col_3_3:
            st.image("exemplos/exemplo1_img.png", caption="Matriz de Similaridades das Habilidades da Caitlyn e do Akshan, cada habilidade é representada por um vetor de 200 dimensões")
        with col_3_4:
            st.image("exemplos/exemplo1_img_pca.png", caption="Matriz de Similaridades das Habilidades da Caitlyn e do Akshan, cada habilidade é representada por um vetor de 20 dimensões ou seja aplicado o PCA")
        st.markdown("---")
        st.markdown(""" <h2 class = "sub_header_func_comp">VÍDEOS DE EXEMPLO HABILIDADES</h2>""",unsafe_allow_html=True)
        col_3_3, col_3_4 = st.columns(2)
        with col_3_3:
            st.video("exemplos/akshan_r.mp4")
        with col_3_4:
            
            st.video("exemplos/caitlyn_r.mp4")
            
else:
    st.error("Não foi possível carregar os dados. Por favor, verifique o arquivo de dados.")


st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Rafa Chaffin</p>
</div>
""", unsafe_allow_html=True)