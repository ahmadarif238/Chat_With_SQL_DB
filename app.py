import streamlit as st
from pathlib import Path
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
import sqlite3

# Streamlit page configuration
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="📊", layout="wide")

st.title("\U0001F4D3 LangChain: Chat with SQL Database")
st.markdown("""Interact with your database seamlessly using natural language queries powered by LangChain and Groq AI.""")

# Sidebar configuration
st.sidebar.title("Database Connection")
st.sidebar.markdown("Configure your database connection below.")

# Database connection options
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

# Sidebar options
radio_opt = ["Use SQLite Database (student.db)", "Connect to MySQL Database"]
selected_opt = st.sidebar.radio(label="Choose the database to connect to:", options=radio_opt)

# Database configuration
mysql_host, mysql_user, mysql_password, mysql_db = None, None, None, None  # Default values
if radio_opt.index(selected_opt) == 1:  # MySQL
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host", placeholder="e.g., localhost")
    mysql_user = st.sidebar.text_input("MySQL User", placeholder="e.g., root")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database Name", placeholder="e.g., test_db")
else:  # SQLite
    db_uri = LOCALDB

# Groq API Key Input
api_key = st.sidebar.text_input(label="Groq API Key", type="password")

# Display error messages if fields are missing
if not db_uri:
    st.sidebar.warning("\U000026A0 Please choose a database connection.")
if not api_key:
    st.sidebar.warning("\U000026A0 Groq API Key is required.")

# Function to configure the database
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        db_filepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

# Initialize the Groq LLM model
if api_key and db_uri:
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    # Chat history
    if "messages" not in st.session_state or st.sidebar.button("Clear Chat History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you?"}]

    # Display chat history
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # User input
    st.markdown("---")
    st.subheader("\U0001F4AC Chat with Your Database")
    user_query = st.chat_input(placeholder="Ask a question about the database...")

    if user_query:
        st.session_state["messages"].append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state["messages"].append({"role": "assistant", "content": response})

            # Display result in a styled format
            st.markdown("### \U0001F4CA Query Result")
            st.code(response)
else:
    st.warning("\U0001F6AB Please provide the required database and API key details to proceed.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Built by Arif Ahmad Khan ❤️</div>", unsafe_allow_html=True)
