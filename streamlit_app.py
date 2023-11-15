import streamlit as st
import pandas as pd
# get the enviroment ready with the API key 
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


def get_completion(prompt, model="gpt-3.5-turbo-1106"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.25, 
    )
    return response.choices[0].message["content"]

# Load your data
# Load data
@st.cache_data  # This will cache the data and won't reload unless the file changes.
def load_data():
    return pd.read_csv('processed_stats_season_p1.csv')

df = load_data() # Replace 'your_file.csv' with your actual file path

# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0 (lower values mean less random answers)
chat = ChatOpenAI(temperature=0.25)

pre_match_template = """\
Act as a professional analyst in football in which your job is to read, understand and compare metrics and statistics of teams, correlate metrics with tactis \
indentify strengths and weaknesses of teams based on statistics and metrics and propose pontetial plans to succeed in the match with the already known outcomes.
Articulate in a professional and appealing manner the analysis of the match and the teams involved, translate complex data into actionable insights and \
recommendations for the team to succeed in the match. At the end think in terms of probability about what's the most likely outcome of the game

Be concise and right into the conclusion of the analysis, do not put an introduction, just go to the conclusion.

Data of both teams: {text}
""" 

prompt_template = ChatPromptTemplate.from_template(pre_match_template)

# Streamlit interface
st.title('Team Statistics Comparison')

# Team selection
team1 = st.selectbox('Select Team 1', options=df['officialName'].unique())
team2 = st.selectbox('Select Team 2', options=df['officialName'].unique())

# Function to get team data
def get_team_data(team_name):
    team_data = df[df['officialName'] == team_name]
    # Example usage for detailed analysis of a single team
    #team_analysis = detailed_team_analysis(team_data, team_name=team_name)

    # Process the data as needed
    return team_data.to_dict() # Example: Convert DataFrame to dictionary

if st.button('Compare'):
    # Display team data
    if team1 and team2:
        team1_data = get_team_data(team1)
        team2_data = get_team_data(team2)
        # If you have a chat function
        customer_messages = prompt_template.format_messages(text={'team1': team1_data, 'team2': team2_data})
        response = chat(customer_messages)
        st.write(response.content)
