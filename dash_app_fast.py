import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import base64
import io
from gRAG_fast import rag_query, feed_document_to_memory, graph

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

chat_history = []

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("⚡ Local GPU RAG Assistant"), width=12)),

    dbc.Row([
        dbc.Col([
            html.H4("Chat Interface"),
            dcc.Textarea(id="chat_input", placeholder="Ask me something...", style={"width":"100%", "height":"70px"}),
            dbc.Button("Send", id="send_btn", color="primary", className="mt-2"),
            dcc.Loading(
                id="loading_chat",
                type="circle",
                children=html.Div(id="chat_history", style={
                    "marginTop":"20px","whiteSpace":"pre-wrap",
                    "height":"400px","overflowY":"scroll",
                    "border":"1px solid #ccc","padding":"10px"
                })
            )
        ], width=6),

        dbc.Col([
            html.H4("Upload Knowledge Files"),
            dcc.Upload(
                id='upload_file',
                children=html.Div(['📄 Drag and Drop or ', html.A('Select PDF/CSV/TXT')]),
                style={
                    'width': '100%',
                    'height': '70px',
                    'lineHeight': '70px',
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

# -------- Chat Callback --------
@app.callback(
    Output("chat_history", "children"),
    Input("send_btn", "n_clicks"),
    State("chat_input", "value")
)
def update_chat(n, message):
    if not n or not message:
        return "\n".join([f"{u}: {m}" for u, m in chat_history])
    chat_history.append(("You", message))
    answer = rag_query(message)
    chat_history.append(("Assistant", answer))
    return "\n".join([f"{u}: {m}" for u, m in chat_history])

# -------- File Upload --------
@app.callback(
    Output('upload_status', 'children'),
    Input('upload_file', 'contents'),
    State('upload_file', 'filename')
)
def upload_file(contents, filename):
    if contents is None:
        return ""
    data = contents.split(",")[1]
    decoded = base64.b64decode(data)
    with open(filename, "wb") as f:
        f.write(decoded)
    feed_document_to_memory(filename)
    return f"✅ Ingested {filename}"

# -------- Graph Refresh --------
@app.callback(
    Output('graph_vis', 'elements'),
    Input('send_btn', 'n_clicks')
)
def update_graph(n):
    elements = []
    for n in graph.nodes():
        label = graph.nodes[n].get("text", n)[:30]
        elements.append({'data': {'id': n, 'label': label}})
    for a, b, d in graph.edges(data=True):
        elements.append({'data': {'source': a, 'target': b, 'label': d.get('relation','relates_to')}})
    return elements

if __name__ == "__main__":
    print("🚀 Running locally at http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050)
