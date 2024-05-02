import streamlit as st
st.set_page_config(page_title = 'Hunome Synthesis prototype', page_icon = '🧊', layout = 'wide', initial_sidebar_state = 'expanded')

# Importing the libraries
import os
import re
import datetime
import hmac
from infisical_client import InfisicalClient, ClientSettings, GetSecretOptions
import pandas as pd
# from graphdatascience import GraphDataScience
from datetime import date
import openai

from sqlalchemy import create_engine, cast, Date
from sqlalchemy.orm import Session, DeclarativeBase, aliased
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import insert, JSONB, BYTEA

from pyvis.network import Network
from tempfile import NamedTemporaryFile
from seaborn import color_palette
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from scipy import spatial
import numpy as np

import instructor
from pydantic import Field, BaseModel, model_validator, FieldValidationInfo
from typing import List
import json

import logging
logger = logging.getLogger(st.__name__)

SPARK_MIN_LIMIT = 5
SPARK_MAX_LIMIT = 40

def check_password():
    '''Returns `True` if the user had the correct password.'''

    def password_entered():
        '''Checks whether a password entered by the user is correct.'''
        if hmac.compare_digest(st.session_state['password'], st.secrets['password']):
            st.session_state['password_correct'] = True
            del st.session_state['password']  # Don't store the password.
        else:
            st.session_state['password_correct'] = False

    # Return True if the passward is validated.
    if st.session_state.get('password_correct', False):
        return True

    # Show input for password.
    st.text_input(
        'Password', type='password', on_change=password_entered, key='password'
    )
    if 'password_correct' in st.session_state:
        st.error('😕 Password incorrect')
    return False

class Base(DeclarativeBase):
    pass

class SparkEmbeddings(Base):
    __tablename__ = "spark_embeddings"
    spark_id = sa.Column(sa.VARCHAR(36), primary_key = True)
    map_id = sa.Column(sa.VARCHAR(36))
    author = sa.Column(sa.String)
    entity_updated = sa.Column(sa.DateTime)
    title = sa.Column(sa.String)
    fulltext = sa.Column(sa.String)
    embedding = sa.Column(Vector(384))
    keywords = sa.Column(JSONB)
    summary = sa.Column(sa.String)
    cluster_id = sa.Column(sa.Integer)
    entity_created = sa.Column(sa.DateTime)
    author_id = sa.Column(sa.VARCHAR(36))
    parent_id = sa.Column(sa.VARCHAR(36))
    sentiment = sa.Column(JSONB)

class Clusters(Base):
    __tablename__ = "clusters"
    cluster_id = sa.Column(sa.Integer, primary_key = True)
    theme = sa.Column(sa.String)
    model = sa.Column(BYTEA)

@st.cache_data
def get_secrets() -> dict[str, str]:
    secrets = {}
    if_client = InfisicalClient(ClientSettings(client_id = os.environ.get('INFISICAL_MACHINE_ID'), client_secret = os.environ.get('INFISICAL_TOKEN')))

    secrets['DWH_USER']   = if_client.getSecret(options = GetSecretOptions(project_id = '651c0e8b857bd029208ead6d', environment = 'dev', secret_name = 'DWH_PG_USER')).secret_value
    secrets['DWH_PW']     = if_client.getSecret(options = GetSecretOptions(project_id = '651c0e8b857bd029208ead6d', environment = 'dev', secret_name = 'DWH_PG_PW')).secret_value
    secrets['DWH_HOST']   = if_client.getSecret(options = GetSecretOptions(project_id = '651c0e8b857bd029208ead6d', environment = 'dev', secret_name = 'DWH_PG_HOST')).secret_value
    secrets['DWH_DBNAME'] = if_client.getSecret(options = GetSecretOptions(project_id = '651c0e8b857bd029208ead6d', environment = 'dev', secret_name = 'DWH_PG_DBNAME')).secret_value

    secrets['TOGETHER_API_KEY'] = if_client.getSecret(options = GetSecretOptions(project_id = '651c0e8b857bd029208ead6d', environment = 'dev', secret_name = 'TOGETHER_API_KEY')).secret_value
    secrets['GOOGLE_API_KEY'] =   if_client.getSecret(options = GetSecretOptions(project_id = '651c0e8b857bd029208ead6d', environment = 'dev', secret_name = 'GOOGLE_API_KEY')).secret_value
    secrets['OPENAI_API_KEY'] =   if_client.getSecret(options = GetSecretOptions(project_id = '651c0e8b857bd029208ead6d', environment = 'dev', secret_name = 'OPENAI_API_KEY')).secret_value

    # secrets['HNM_NEO4J_HOST'] = if_client.get_secret('HNM_NEO4J_HOST').secret_value
    # secrets['HNM_NEO4J_USER'] = if_client.get_secret('HNM_NEO4J_USER').secret_value
    # secrets['HNM_NEO4J_PW']   = if_client.get_secret('HNM_NEO4J_PASSWORD').secret_value

    return secrets

@st.cache_data(ttl = '1d')
def get_sparkmaps() -> pd.DataFrame:
    secrets = get_secrets()
    engine = create_engine(f'postgresql+psycopg2://{secrets["DWH_USER"]}:{secrets["DWH_PW"]}@{secrets["DWH_HOST"]}:5432/{secrets["DWH_DBNAME"]}')
    with Session(engine) as session:
        maps_base = aliased(SparkEmbeddings)
        maps_title = aliased(SparkEmbeddings)
        sparkmaps = session.query(maps_base.map_id, sa.func.count(), maps_title.title).\
            filter(maps_base.map_id == maps_title.map_id, maps_title.map_id == maps_title.spark_id).\
            group_by(maps_base.map_id, maps_title.title).having(sa.func.count() >= 5).order_by(sa.func.count().desc()).all()
        sparkmaps_df = pd.DataFrame(sparkmaps, columns=['map_id', 'count', 'title'])
        return sparkmaps_df

@st.cache_data(ttl = '1d')
def get_sparkmap_dates(map_id: str) -> list[date]:
    secrets = get_secrets()
    engine = create_engine(f'postgresql+psycopg2://{secrets["DWH_USER"]}:{secrets["DWH_PW"]}@{secrets["DWH_HOST"]}:5432/{secrets["DWH_DBNAME"]}')
    with Session(engine) as session:
        sparkmap_dates = session.query(cast(SparkEmbeddings.entity_updated, Date)).filter(
            SparkEmbeddings.map_id == map_id
        ).distinct().all()
        dates = [row[0] for row in sparkmap_dates]
        dates.sort()
        return dates

@st.cache_data(ttl = '1d')
def get_sparks(
    map_id: str,
    # min_date: date, max_date: date
) -> pd.DataFrame:
    secrets = get_secrets()
    engine = create_engine(
        f'postgresql+psycopg2://{secrets["DWH_USER"]}:{secrets["DWH_PW"]}@{secrets["DWH_HOST"]}:5432/{secrets["DWH_DBNAME"]}',
        echo = True
    )
    with Session(engine) as session:
        statement = session.query(SparkEmbeddings, Clusters.theme).\
            join(Clusters, Clusters.cluster_id == SparkEmbeddings.cluster_id, isouter = True).\
            filter(SparkEmbeddings.map_id == map_id)
        sparks_df = pd.read_sql(statement.statement, statement.session.bind)
        logger.info('Sparks shape: ' + str(sparks_df.shape))
        logger.info('Sparks columns: ' + str(sparks_df.columns.tolist()))
        sparks_df['entity_updated'] = pd.to_datetime(sparks_df['entity_updated'])
        sparks_df['entity_created'] = pd.to_datetime(sparks_df['entity_created'])
        return sparks_df

class Idea(BaseModel):
    idea: str = Field(..., title = 'Idea')
    article_id: str = Field(..., title = 'ID of the article')
    author: str = Field(..., title = 'Author of the article')

class ExtractedIdeas(BaseModel):
    """Extracted ideas from a set of articles"""
    ideas: List[Idea] = Field(..., title = 'Extracted ideas')

def build_context(sparks: pd.DataFrame) -> str:
    context = []
    for _, row in sparks.iterrows():
        context.append({
            'id': row['spark_id'],
            'title': row['title'],
            'text': row['fulltext'],
            'author': row['author_id'],
            'parent_id': row['parent_id'],
            'created': row['entity_created'].strftime('%Y-%m-%d %H:%M:%S'),
        })
    return json.dumps(context)

@st.cache_data
def get_synthesis(sparks: pd.DataFrame, model:str) -> dict:
    secrets = get_secrets()
    client = instructor.patch(openai.OpenAI(api_key=secrets['OPENAI_API_KEY']))
    ideas: ExtractedIdeas = client.chat.completions.create(
        model=model,
        response_model=ExtractedIdeas,
        messages=[
            {"role": "system", "content": """
You are a system that extracts main ideas from a set of articles. You must use only the provided context and extract up to 10 ideas.\
You do observe the hierarchy of the articles indicated by parent_id and their chronological order."""},
            {"role": "user", "content": build_context(sparks)}
        ]
    )
    logger.info('Tokens in response: ' + str(ideas._raw_response.usage))
    return ideas.ideas


def generate_synthesys(sparks: pd.DataFrame, model: str) -> str:
    unique_dates = sparks['entity_updated'].dt.date.unique().shape[0]
    sparks_in_cluster = sparks.groupby('cluster_id').count()['spark_id'].reset_index()
    if sparks.shape[0] < 5:
        return f'Too few sparks for synthesis. Found {sparks.shape[0]} sparks in {sparks.cluster_id.unique().shape[0]} clusters ({"/".join(sparks.groupby("cluster_id").count()["spark_id"].astype(str).values.tolist())}).'
    elif sparks_in_cluster.spark_id.max() > 50 and sparks['entity_updated'].dt.date.unique().shape[0] > 1:
        return f'Too many sparks for synthesis. Found {sparks.shape[0]} sparks in {sparks.cluster_id.unique().shape[0]} clusters ({"/".join(sparks.groupby("cluster_id").count()["spark_id"].astype(str).values.tolist())}).'
    if sparks.shape[0] > SPARK_MAX_LIMIT and sparks_in_cluster.shape[0] > 1:
        synthesis = ''
        for cluster in sparks.cluster_id.unique():
            if sparks[sparks['cluster_id'] == cluster].empty:
                continue
            cluster_theme = sparks[sparks['cluster_id'] == cluster]['theme'].values[0]
            cluster_sparks = sparks[sparks['cluster_id'] == cluster]
            cluster_synthesis = get_synthesis(cluster_sparks, model)
            synthesis += f'**Cluster: {cluster_theme}**\n\n{parse_response(cluster_synthesis, cluster_sparks)}\n\n'
        return synthesis
    else:
        return parse_response(get_synthesis(sparks, model), sparks)



def parse_response(response: str, sparks: pd.DataFrame) -> str:
    logger.info('Response length: ' + str(len(response)))
    # logger.info('Response: ' + str(response))
    # response = re.sub(replace_pattern, f'[{author_name}](https://platform.hunome.com/profile/{author[1]})', response)
    # response = re.sub(replace_pattern, f'[{spark_title}](https://platform.hunome.com/sparkmap/view-spark/{spark})', response)
    parsed_response = ''
    i = 1
    for idea in response:
        author_name = sparks[sparks['author_id'] == idea.author]['author'].values[0] if not sparks[sparks['author_id'] == idea.author].empty else idea.author
        spark_title = sparks[sparks['spark_id'] == idea.article_id]['title'].values[0] if not sparks[sparks['spark_id'] == idea.article_id].empty else idea.article_id
        parsed_response += f'1. {idea.idea} ([{spark_title}](https://platform.hunome.com/sparkmap/view-spark/{idea.article_id}) by [{author_name}](https://platform.hunome.com/profile/{idea.author}))\n\n'
        i += 1

    # logger.info('Parsed response: ' + str(parsed_response))
    return parsed_response



def format_spark_map_select(sparkmaps: dict[str, list[str, int]], sparkmap_id: str) -> str:
    sparkmap = sparkmaps[sparkmaps['map_id'] == sparkmap_id]
    return f'{sparkmap["title"].values[0]} (#{sparkmap["count"].values[0]})'


def draw_sparkmap(sparks: pd.DataFrame, color_type: str) -> str:
    cmap = LinearSegmentedColormap.from_list('rg',["g", "y", "r"], N = 256)
    pallete = color_palette('pastel', n_colors = sparks.sort_values('cluster_id').cluster_id.unique().shape[0]).as_hex()
    pallete = {cluster: pallete[i] for i, cluster in enumerate(sparks.cluster_id.unique())}
    if color_type == 'Heat':
        sparks['positive_sentiment'] = sparks.sentiment.apply(lambda x: [y['score'] for y in x if y['label'] == 'positive'][0])
        sparks = sparks.merge(
            sparks[['spark_id', 'embedding']].rename(columns = {'embedding': 'parent_embedding', 'spark_id': 'parent_id'}),
            on = 'parent_id', how = 'left'
        )
        sparks['cosine_similarity'] = sparks.apply(
            lambda x: 1 - spatial.distance.cosine(x['embedding'], x['parent_embedding']) if isinstance(x['parent_embedding'], np.ndarray) else 1,
            axis = 1
        )
    light_grey = '#d3d3d3'
    net = Network(height = '800px', width = '100%')
    net.set_options("""
    var options = {
                    "layout": {
                        "randomSeed": 42
                    }
                }
    """)
    def get_color(row, color_type):
        if not row['is_selected']:
            return light_grey
        if color_type == 'Heat':
            heat = np.log(np.exp(1 - row['positive_sentiment']) * np.exp(1 - row['cosine_similarity'])) / 2
            return rgb2hex(cmap(heat))
        if not row['cluster_id'] or np.isnan(row['cluster_id']):
            return light_grey
        # logger.info(f'Cluster id {row["cluster_id"]}')
        return pallete[row['cluster_id']]
    
    for _, row in sparks.iterrows():
        # if row['spark_id'] == 'c79720c0-bf65-4fec-af65-045af831b3c3':
        #     continue
        # logger.info(f'Adding node {row["spark_id"]} with color {get_color(row, color_type)}')
        net.add_node(
            row['spark_id'],
            label = row['title'][:20],
            color = get_color(row, color_type),
            title = f'''
            Title: {row['title'][:20]}
            Author: {row['author']}
            '''
        )
    for _, row in sparks.iterrows():
        # if row['spark_id'] == 'c79720c0-bf65-4fec-af65-045af831b3c3':
        #     continue
        if row['parent_id'] and row['parent_id'] in sparks.spark_id.values:
            net.add_edge(row['parent_id'], row['spark_id'])
    # Add box shaped nodes with cluster titles
    step = 50
    x = -1200
    y = -800
    for cluster_id in sparks.cluster_id.unique():
        if not cluster_id or np.isnan(cluster_id):
            continue
        # logger.info(f'Adding cluster {cluster_id}')
        cluster_title = f'Cluster {sparks[sparks["cluster_id"] == cluster_id]["theme"].values[0]} ({sparks[sparks["cluster_id"] == cluster_id].shape[0]})'
        net.add_node(
            cluster_title,
            shape = 'box',
            color = pallete[cluster_id],
            title = cluster_title,
            x = x,
            y = y,
            physics = False,
            value = 80
        )
        y = y + step
    
    with NamedTemporaryFile(mode = 'w', suffix = '.html') as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as f:
            return '\n'.join(f.readlines())


if __name__ == '__main__':
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.
    sparkmaps = get_sparkmaps()
    model = st.sidebar.selectbox(
        'Select model',
        ['gpt-3.5-turbo', 'gpt-4-turbo']
    )
    color_type = st.sidebar.selectbox(
        'Select color type',
        ['Cluster', 'Heat']
    )
    sparkmap_id = st.sidebar.selectbox(
        'Select Spark Map',
        sparkmaps['map_id'].tolist(),
        format_func = lambda x: format_spark_map_select(sparkmaps, x)
    )
    sparkmap_dates = get_sparkmap_dates(sparkmap_id)
    # logger.info('Sparkmap dates: ' + str(sparkmap_dates))
    time_interval = st.sidebar.select_slider(
        'Time interval',
        options = sparkmap_dates,
        value = (
            sparkmap_dates[0],
            sparkmap_dates[-1]
        )
    )
    sparks = get_sparks(sparkmap_id)
    sparks['is_selected'] = (sparks['entity_updated'].dt.date >= time_interval[0]) & (sparks['entity_updated'].dt.date <= time_interval[1])
    selected_sparks = sparks[sparks['is_selected']]
    sparks_in_cluster = selected_sparks.groupby('cluster_id').count()['spark_id'].reset_index()
    color = 'green'
    if selected_sparks.shape[0] < 5:
        color = 'red'
    elif sparks_in_cluster.spark_id.max() > 50 and selected_sparks['entity_updated'].dt.date.nunique() > 1:
        color = 'orange'
    st.sidebar.write(f':{color}[Found {selected_sparks.shape[0]} sparks in {selected_sparks.cluster_id.nunique()} clusters ({"/".join(sparks_in_cluster.spark_id.astype(str).values.tolist())}).]')

    sparkmap_vis = draw_sparkmap(sparks, color_type)
    with st.expander('Spark Map visualization', expanded = True):
        st.components.v1.html(sparkmap_vis, height = 820)

    if st.sidebar.button('Generate synthesis'):
        synthesis = generate_synthesys(selected_sparks, model)
        st.write(synthesis)

    if st.sidebar.button("Clear synthesis cache"):
        get_synthesis.clear()
    st.sidebar.markdown('---')
    st.sidebar.markdown(f'''
    **NB!** Minimum number of sparks for synthesis is {SPARK_MIN_LIMIT}.
    Maximum number of sparks for synthesis is {SPARK_MAX_LIMIT} per cluster.
    If you have more than {SPARK_MAX_LIMIT} sparks, please reduce the time interval.
    If you have less than {SPARK_MIN_LIMIT} sparks, please extend the time interval.
    ''')
    st.sidebar.markdown('**NB!** This is a prototype. The synthesis is generated by an AI model and may not be accurate. Please use it with caution.')
    st.sidebar.markdown('---')
    st.sidebar.write(f'© {datetime.date.today().year} Hunome')
