# Model 1 + Model 2 Integration

`conversation_pipeline.py` runs the conversation loop:

1. Model 1 opens the conversation.
2. The patient enters a reply.
3. Model 2 predicts the next dialogue action, such as `pain_location` or `RED_FLAG`.
4. Model 1 converts that action into one natural-language response.

Model 2's label is used as an internal instruction and is not added to the dialogue history sent to the patient. The same history is passed to Model 2 on every turn.

## Run

From the repository root, activate the project environment and run:

```powershell
.\.venv\Scripts\Activate.ps1
python ".\model1 and 2 integartion\conversation_pipeline.py"
```

The runner requires a CUDA-enabled environment for the 4-bit Model 1 setup. The first run may also download `all-mpnet-base-v2` and requires access to the base Llama model used by the adapter.

Type `exit`, `quit`, or `bye` to stop the conversation.

## HTTP API

Start the API from this directory on the laptop that has the models:

```powershell
cd ".\model1 and 2 integartion"
..\.venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The server laptop and the client laptop must be on the same network. Find the
server laptop's local IPv4 address with `ipconfig`. The client can then open:

```text
http://SERVER_IP:8000/docs
```

Start a conversation:

```http
POST http://SERVER_IP:8000/conversation/start
Content-Type: application/json

{"patient_id":"demo-user"}
```

Send the patient's next message using the returned `session_id`:

```http
POST http://SERVER_IP:8000/conversation/SESSION_ID/message
Content-Type: application/json

{"message":"I have had a fever since yesterday."}
```

The API keeps conversation history in server memory. A restart clears all
sessions. The server requires a CUDA-enabled machine for Model 1 inference.