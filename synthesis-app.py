import streamlit as st
st.set_page_config(page_title = 'Hunome RAG prototype', page_icon = '🧊', layout = 'wide', initial_sidebar_state = 'expanded')

# Importing the libraries
import os
import re
import hmac
from infisical import InfisicalClient
import pandas as pd
# from graphdatascience import GraphDataScience
from datetime import date
import openai

from sqlalchemy import create_engine, cast, Date
from sqlalchemy.orm import Session, DeclarativeBase, aliased
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import insert, JSONB, BYTEA

import logging
@st.cache_data
def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

logger = get_logger()

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

class Clusters(Base):
    __tablename__ = "clusters"
    cluster_id = sa.Column(sa.Integer, primary_key = True)
    theme = sa.Column(sa.String)
    model = sa.Column(BYTEA)

@st.cache_data
def get_secrets() -> dict[str, str]:
    secrets = {}
    if_client = InfisicalClient(token = os.environ.get('INFISICAL_TOKEN'))

    secrets['DWH_USER']   = if_client.get_secret('DWH_PG_USER').secret_value
    secrets['DWH_PW']     = if_client.get_secret('DWH_PG_PW').secret_value
    secrets['DWH_HOST']   = if_client.get_secret('DWH_PG_HOST').secret_value
    secrets['DWH_DBNAME'] = if_client.get_secret('DWH_PG_DBNAME').secret_value

    secrets['TOGETHER_API_KEY'] = if_client.get_secret('TOGETHER_API_KEY').secret_value

    # secrets['HNM_NEO4J_HOST'] = if_client.get_secret('HNM_NEO4J_HOST').secret_value
    # secrets['HNM_NEO4J_USER'] = if_client.get_secret('HNM_NEO4J_USER').secret_value
    # secrets['HNM_NEO4J_PW']   = if_client.get_secret('HNM_NEO4J_PASSWORD').secret_value

    return secrets

@st.cache_data
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

@st.cache_data
def get_sparkmap_dates(map_id: str) -> list[date]:
    secrets = get_secrets()
    engine = create_engine(f'postgresql+psycopg2://{secrets["DWH_USER"]}:{secrets["DWH_PW"]}@{secrets["DWH_HOST"]}:5432/{secrets["DWH_DBNAME"]}')
    with Session(engine) as session:
        sparkmap_dates = session.query(cast(SparkEmbeddings.entity_created, Date)).filter(
            SparkEmbeddings.map_id == map_id
        ).distinct().all()
        dates = [row[0] for row in sparkmap_dates]
        dates.sort()
        return dates

@st.cache_data
def get_sparks(map_id: str, min_date: date, max_date: date) -> pd.DataFrame:
    secrets = get_secrets()
    engine = create_engine(
        f'postgresql+psycopg2://{secrets["DWH_USER"]}:{secrets["DWH_PW"]}@{secrets["DWH_HOST"]}:5432/{secrets["DWH_DBNAME"]}',
        echo = True
    )
    with Session(engine) as session:
        statement = session.query(SparkEmbeddings).filter(
            SparkEmbeddings.map_id == map_id,
            SparkEmbeddings.entity_created >= min_date,
            SparkEmbeddings.entity_created <= max_date
        )
        sparks_df = pd.read_sql(statement.statement, statement.session.bind)
        logger.debug('Sparks shape: ' + str(sparks_df.shape))
        logger.debug('Sparks columns: ' + str(sparks_df.columns.tolist()))
        sparks_df['entity_updated'] = pd.to_datetime(sparks_df['entity_updated'])
        sparks_df['entity_created'] = pd.to_datetime(sparks_df['entity_created'])
        return sparks_df


@st.cache_data
def generate_synthesys(sparks: pd.DataFrame, model: str) -> str:
    unique_dates = sparks['entity_updated'].dt.date.unique().shape[0]
    if sparks.shape[0] == 0:
        return 'No sparks found. Please extend the time interval.'
    if sparks.shape[0] < 5:
        return f'Not enough sparks found ({sparks.shape[0]}). Please extend the time interval.'
    if sparks.shape[0] > 50 and unique_dates > 1:
        return f'Too many sparks found ({sparks.shape[0]}). Please reduce the time interval.'
    st.write(f'Found {sparks.shape[0]} sparks.')
    prompt = '''
You will extract a summary of discussion from the list of sparks below.
You will use exact citations from the sparks to support the summary, mentioning the author with name and ID as {Author:Name;author-id} and spark as {Spark:spark-id}.
You should constrain your synthesis to minimum 5 and maximum 10 key ideas discussed.
You should not introduce new ideas, but only summarize the existing ones.
Your response should be like a recap of discussion, taking into account the chronology of the sparks. Each spark may have a parent within the given set, outside it or not set. Is parent ID is set it means that the given Spark is a response to the parent spark.
Please do not add the date and parent of the spark, but use the order of the sparks to determine the chronology.
Please write the response as an essay, not as a list of citations.
Here is an example respones, please make sure you strictly follow the format:

{Author:John Doe;john-doe-id} stated that The world is flat [{Spark:spark-1-id}]. However, {Author:Jane Doe;jane-doe-id} disagreed and said that The world is round [{Spark:spark-2-id}]. {Author:Harry Kane;harry-kane-id} summarized that The world is round, but it is flat in some places[{Spark:spark-3-id}].

You can use the following sparks to generate your response:
{{SPARKS}}
    '''
    prompt = prompt.replace('{{SPARKS}}', '\n'.join(
        [f'''
[SPARK]
    [ID]{row["spark_id"]}[/ID]
    [AUTHOR][NAME]{row["author"]}[/NAME][ID]{row["author_id"]}[/ID][/AUTHOR]
    [PARENT]{row["parent_id"]}[/PARENT]
    [CREATED]{row["entity_created"]}[/CREATED]
    [TITLE]{row["title"]}[/TITLE]
    [TEXT]{row["fulltext"]}[/TEXT]
[/SPARK]''' for _, row in sparks.iterrows()]
    ))
    logger.debug('Prompt length: ' + str(len(prompt)))
    secrets = get_secrets()
    client = openai.OpenAI(
        api_key = secrets['TOGETHER_API_KEY'],
        base_url='https://api.together.xyz',
    )
    response = client.chat.completions.create(
        model = f'mistralai/{model}',
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]
    )
    return response.choices[0].message.content



def parse_response(response: str, sparks: pd.DataFrame) -> str:
    logger.debug('Response length: ' + str(len(response)))
    author_pattern = re.compile(r'[\{\(\[]Author:\s*(?P<name>.*?);(?P<id>.*?)[\}\)\]]')
    spark_pattern = re.compile(r'[\{\(\[]Spark:\s*(?P<id>.*?)[\}\)\]]')
    all_authors = author_pattern.findall(response)
    all_sparks = spark_pattern.findall(response)
    for author in all_authors:
        author_name = sparks[sparks['author_id'] == author[1]]['author'].values[0]
        replace_pattern = re.compile(r'[\{\(\[]Author:\s*[\}\)\]]' + re.escape(author[0]) + r';' + re.escape(author[1]) + r'[\}\)\]]')
        response = re.sub(replace_pattern, f'[{author_name}](https://platform.hunome.com/profile/{author[1]})', response)
    for spark in all_sparks:
        spark_title = sparks[sparks['spark_id'] == spark]['title'].values[0]
        logger.debug(f'Spark {spark} title: {spark_title}')
        replace_pattern = re.compile(r'[\{\(\[]Spark:\s*' + re.escape(spark) + r'[\}\)\]]')
        response = re.sub(replace_pattern, f'[{spark_title}](https://platform.hunome.com/sparkmap/view-spark/{spark})', response)
    return response


def format_spark_map_select(sparkmaps: dict[str, list[str, int]], sparkmap_id: str) -> str:
    sparkmap = sparkmaps[sparkmaps['map_id'] == sparkmap_id]
    return f'{sparkmap["title"].values[0]} (#{sparkmap["count"].values[0]}'

if __name__ == '__main__':
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.
    sparkmaps = get_sparkmaps()
    model = st.sidebar.selectbox(
        'Select model',
        ['Mixtral-8x7B-Instruct-v0.1', 'Mistral-7B-Instruct-v0.2']
    )
    sparkmap_id = st.sidebar.selectbox(
        'Select Spark Map',
        sparkmaps['map_id'].tolist(),
        format_func = lambda x: format_spark_map_select(sparkmaps, x)
    )
    sparkmap_dates = get_sparkmap_dates(sparkmap_id)
    logger.debug('Sparkmap dates: ' + str(sparkmap_dates))
    time_interval = st.sidebar.select_slider(
        'Time interval',
        options = sparkmap_dates,
        value = (
            sparkmap_dates[0],
            sparkmap_dates[-1]
        )
    )
    if st.sidebar.button('Generate synthesis'):
        sparks = get_sparks(sparkmap_id, time_interval[0], time_interval[1])
        st.write(parse_response(generate_synthesys(sparks, model), sparks))
