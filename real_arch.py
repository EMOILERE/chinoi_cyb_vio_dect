import torch
from transformers import BertTokenizer, BertForSequenceClassification
from graphviz import Digraph

tokenizer = BertTokenizer.from_pretrained(r'E:\\chinese_senti\\sentiment_tokenizer_finetuned3')
model = BertForSequenceClassification.from_pretrained(r'E:\\chinese_senti\\sentiment_model_finetuned3')

dummy_input_ids = torch.zeros(1, 128, dtype=torch.long)
dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

def add_nodes(var, dot=None, parent=None):
    if dot is None:
        dot = Digraph(format='png', graph_attr={'rankdir': 'TB', 'nodesep': '1', 'ranksep': '2'})

    node_id = str(id(var))

    dot.node(node_id, type(var).__name__)

    if parent is not None:
        dot.edge(str(id(parent)), node_id)

    if hasattr(var, 'named_children'):
        for name, module in var.named_children():
            with dot.subgraph(name=f'cluster_{name}') as sub:
                sub.attr(label=name, style='dotted')
                add_nodes(module, sub, var)

    return dot

dot = add_nodes(model)
dot.render('model_architecture4')

print("模型架构图已保存为 model_architecture4.png")
