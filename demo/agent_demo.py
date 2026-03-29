import pprint
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# pip install -qU deepagents
from deepagents import create_deep_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


research_instructions = """\
You are a helpful assistant
"""


agent = create_deep_agent(
    tools=[get_weather],
    system_prompt=research_instructions,
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
pprint.pprint(result)
