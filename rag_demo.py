import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

class SimpleRAG:
    def __init__(self):
        # 初始化嵌入模型（用于文档检索）
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 初始化生成模型（轻量级）
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.generator = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
        
        # 知识库
        self.documents = []
        self.doc_embeddings = None
        
        # 默认知识库内容
        self.load_default_knowledge()
    
    def load_default_knowledge(self):
        """加载默认知识库"""
        default_docs = [
            "大语言模型（LLM）是基于Transformer架构的深度学习模型，能够理解和生成人类语言。",
            "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的方法，通过检索相关文档来增强生成质量。",
            "Transformer是一种基于注意力机制的神经网络架构，由Vaswani等人在2017年提出。",
            "BERT是双向编码器表示模型，擅长理解任务如分类、问答等。",
            "GPT是生成式预训练模型，采用自回归方式生成文本。",
            "微调（Fine-tuning）是在预训练模型基础上，使用特定任务数据进行进一步训练的过程。",
            "向量数据库用于存储和检索高维向量，常用于相似度搜索。",
            "提示工程（Prompt Engineering）是设计有效输入提示来引导模型产生期望输出的技术。",
            "量化技术可以将模型权重从32位浮点数压缩到8位或4位整数，显著减少内存使用。",
            "推理优化包括模型量化、剪枝、知识蒸馏等技术，用于提升模型部署效率。"
        ]
        self.add_documents(default_docs)
    
    def add_documents(self, docs):
        """添加文档到知识库"""
        self.documents.extend(docs)
        # 计算文档嵌入
        embeddings = self.embedding_model.encode(docs)
        
        if self.doc_embeddings is None:
            self.doc_embeddings = embeddings
        else:
            self.doc_embeddings = np.vstack([self.doc_embeddings, embeddings])
    
    def retrieve_documents(self, query, top_k=2):
        """检索最相关的文档"""
        if not self.documents:
            return []
        
        # 计算查询嵌入
        query_embedding = self.embedding_model.encode([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        # 获取top_k最相似的文档
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_docs = []
        for idx in top_indices:
            retrieved_docs.append({
                'content': self.documents[idx],
                'similarity': similarities[idx]
            })
        
        return retrieved_docs
    
    def generate_answer(self, query, retrieved_docs):
        """基于检索到的文档生成答案"""
        # 构建上下文
        context = "\n".join([doc['content'] for doc in retrieved_docs])
        
        # 构建提示
        prompt = f"""基于以下上下文信息回答问题：

上下文：
{context}

问题：{query}

答案："""
        
        # 由于DialoGPT主要用于对话，这里我们简化处理
        # 实际项目中建议使用专门的QA模型
        if retrieved_docs:
            # 基于检索到的文档构建简单回答
            answer = self.simple_answer_generation(query, retrieved_docs)
        else:
            answer = "抱歉，我在知识库中没有找到相关信息。"
        
        return answer
    
    def simple_answer_generation(self, query, retrieved_docs):
        """简单的答案生成逻辑"""
        # 找到最相关的文档
        best_doc = retrieved_docs[0]['content']
        
        # 简单的关键词匹配和答案构建
        if "什么是" in query or "是什么" in query:
            return f"根据我的知识库：{best_doc}"
        elif "如何" in query or "怎么" in query:
            related_info = [doc['content'] for doc in retrieved_docs]
            return f"关于您的问题，相关信息如下：\n" + "\n".join(related_info)
        else:
            return f"根据相关资料：{best_doc}\n\n补充信息：{retrieved_docs[1]['content'] if len(retrieved_docs) > 1 else ''}"

# 初始化RAG系统
rag_system = SimpleRAG()

def add_document_interface(doc_text):
    """添加文档的接口"""
    if doc_text.strip():
        # 按句子分割文档
        sentences = re.split(r'[。！？\n]', doc_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        rag_system.add_documents(sentences)
        return f"已添加 {len(sentences)} 个文档片段到知识库"
    return "请输入有效的文档内容"

def rag_chat(query, history):
    """RAG问答主函数"""
    if not query.strip():
        return history, history
    
    # 检索相关文档
    retrieved_docs = rag_system.retrieve_documents(query, top_k=2)
    
    # 生成答案
    answer = rag_system.generate_answer(query, retrieved_docs)
    
    # 添加检索信息
    retrieval_info = ""
    if retrieved_docs:
        retrieval_info = f"\n\n📚 **检索到的相关文档：**\n"
        for i, doc in enumerate(retrieved_docs, 1):
            retrieval_info += f"{i}. {doc['content'][:100]}... (相似度: {doc['similarity']:.3f})\n"
    
    full_answer = answer + retrieval_info
    
    # 更新对话历史
    history.append([query, full_answer])
    
    return history, history

def clear_knowledge_base():
    """清空知识库"""
    global rag_system
    rag_system = SimpleRAG()  # 重新初始化，会加载默认知识
    return "知识库已重置为默认内容"

def show_knowledge_base():
    """显示当前知识库内容"""
    if rag_system.documents:
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(rag_system.documents)])
        return f"当前知识库包含 {len(rag_system.documents)} 个文档：\n\n{docs_text}"
    return "知识库为空"

# 创建Gradio界面
with gr.Blocks(title="🤖 RAG问答系统演示", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 RAG问答系统演示
    
    这是一个基于检索增强生成（RAG）的问答系统原型。系统会从知识库中检索相关信息，然后生成答案。
    
    ## 功能特点：
    - 📖 知识库管理：添加自定义文档
    - 🔍 智能检索：基于语义相似度检索相关文档  
    - 💬 问答生成：结合检索结果生成答案
    - 📊 透明度：显示检索过程和相似度分数
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📚 知识库管理")
            
            doc_input = gr.Textbox(
                label="添加文档",
                placeholder="输入要添加到知识库的文档内容...",
                lines=5
            )
            
            with gr.Row():
                add_btn = gr.Button("➕ 添加文档", variant="primary")
                clear_btn = gr.Button("🗑️ 重置知识库", variant="secondary")
                show_btn = gr.Button("👁️ 查看知识库", variant="secondary")
            
            doc_status = gr.Textbox(label="状态", interactive=False)
            
            knowledge_display = gr.Textbox(
                label="知识库内容",
                lines=10,
                interactive=False,
                visible=False
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 RAG问答")
            
            chatbot = gr.Chatbot(
                label="对话历史",
                height=400,
                show_label=True
            )
            
            msg = gr.Textbox(
                label="输入问题",
                placeholder="请输入您的问题，比如：什么是RAG？",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 提问", variant="primary")
                clear_chat_btn = gr.Button("🗑️ 清空对话", variant="secondary")
    
    # 示例问题
    gr.Markdown("""
    ### 💡 示例问题：
    - 什么是大语言模型？
    - RAG是什么技术？
    - 如何进行模型微调？
    - Transformer架构有什么特点？
    """)
    
    # 事件绑定
    add_btn.click(
        add_document_interface,
        inputs=[doc_input],
        outputs=[doc_status]
    ).then(
        lambda: "",
        outputs=[doc_input]
    )
    
    clear_btn.click(
        clear_knowledge_base,
        outputs=[doc_status]
    )
    
    show_btn.click(
        show_knowledge_base,
        outputs=[knowledge_display]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[knowledge_display]
    )
    
    submit_btn.click(
        rag_chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, chatbot]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    msg.submit(
        rag_chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, chatbot]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    clear_chat_btn.click(
        lambda: [],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch()