import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import base64
from gRAG_imp1 import rag_query, feed_document_to_memory, graph

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# In-memory chat history
chat_memory = []

# ----------------- Layout -----------------
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("🧠 Local RAG Assistant"), width=12)),

    dbc.Row([
        # Chat Column
        dbc.Col([
            html.H4("Chat"),
            dcc.Textarea(id="chat_input", placeholder="Type a message...", style={"width":"100%", "height":"50px"}),
            dbc.Button("Send", id="send_btn", color="primary", className="mt-2"),
            html.Div(id="chat_history", style={"marginTop":"20px", "whiteSpace":"pre-wrap", "height":"400px", "overflowY":"scroll", "border":"1px solid #ccc", "padding":"10px"})
        ], width=6),

        # Upload & Graph Column
        dbc.Col([
            html.H4("Upload PDF/CSV"),
            dcc.Upload(
                id='upload_file',
                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center'
                },
                multiple=False
            ),
            html.Div(id='upload_status', style={"marginTop":"10px"}),
            html.Hr(),
            html.H5("GraphRAG Snapshot"),
            cyto.Cytoscape(
                id='graph_vis',
                layout={'name': 'cose'},
                style={'width': '100%', 'height': '400px'},
                elements=[]
            )
        ], width=6)
    ])
], fluid=True)

# ----------------- Callbacks -----------------

@app.callback(
    Output("chat_history", "children"),
    Input("send_btn", "n_clicks"),
    State("chat_input", "value")
)
def update_chat(n_clicks, message):
    if not message:
        return "\n".join([f"{r}: {m}" for r, m in chat_memory])
    
    # Query your gRAG_imp2 RAG system
    answer = rag_query(message)
    
    chat_memory.append(("You", message))
    chat_memory.append(("Assistant", answer))
    
    return "\n".join([f"{r}: {m}" for r, m in chat_memory])

@app.callback(
    Output('upload_status', 'children'),
    Input('upload_file', 'contents'),
    State('upload_file', 'filename')
)
def upload_file(contents, filename):
    if contents is None:
        return ""
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Save file locally
    with open(filename, "wb") as f:
        f.write(decoded)
    
    # Feed into RAG memory
    feed_document_to_memory(filename)
    
    return f"✅ Ingested {filename}"

@app.callback(
    Output('graph_vis', 'elements'),
    Input('send_btn', 'n_clicks')
)
def update_graph(n_clicks):
    elements = []
    for n in graph.nodes():
        label = graph.nodes[n].get("text", n)[:20]  # show short label
        elements.append({'data': {'id': n, 'label': label}})
    for a, b, d in graph.edges(data=True):
        elements.append({'data': {'source': a, 'target': b, 'label': d.get('relation','relates_to')}})
    return elements

# ----------------- Run server -----------------
if __name__ == "__main__":
    print("Running locally at http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050)

