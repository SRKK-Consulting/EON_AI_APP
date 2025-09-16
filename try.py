from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

# Create a client pointing to your Azure AI Foundry project
project = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint="https://aiserver1234.services.ai.azure.com/api/projects/SALES_CHATBOT_POC"
)

# Get the agent by ID
agent = project.agents.get_agent("asst_ZkzK0inGhhkrf5NFXOsHetRU")

# Create a conversation thread
thread = project.agents.threads.create()
print(f"✅ Created thread, ID: {thread.id}")

# Send a user message
project.agents.messages.create(
    thread_id=thread.id,
    role="user",
    content="Hi Agent158. Tell me the weather forecast in kuala lumpur on 10 Sept 2025"
)

# Run the agent
run = project.agents.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id
)

# Check run status
if run.status == "failed":
    print(f"❌ Run failed: {run.last_error}")
else:
    print("✅ Run completed successfully. Conversation so far:\n")

    # Fetch all messages in order
    messages = project.agents.messages.list(
        thread_id=thread.id,
        order=ListSortOrder.ASCENDING
    )

    for message in messages:
        if message.text_messages:
            text = message.text_messages[-1].text.value
            print(f"{message.role}: {text}")

