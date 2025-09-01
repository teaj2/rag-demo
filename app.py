import gradio as gr
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import jieba  # 中文分词

class SimpleRAG:
    def __init__(self):
        # 使用TF-IDF作为轻量级的文本向量化方法
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 3),  # 增加3-gram提高匹配度
            min_df=1,  # 最小文档频率
            lowercase=True,
            analyzer='word'
        )
        
        # 知识库
        self.documents = []
        self.doc_vectors = None
        self.vectorizer_fitted = False
        
        # 加载默认知识库
        self.load_default_knowledge()
    
    def preprocess_text(self, text):
        """文本预处理 - 改进中文处理"""
        # 保留中文字符，移除标点符号
        text = re.sub(r'[^\u4e00-\u9fa5\w\s]', ' ', text)
        # 使用jieba分词处理中文
        try:
            words = jieba.lcut(text)
            return ' '.join(words).lower()
        except:
            return text.lower()
    
    def load_default_knowledge(self):
        """加载默认知识库"""
        default_docs = [
            "大语言模型LLM是基于Transformer架构的深度学习模型能够理解和生成人类语言",
            "RAG检索增强生成是一种结合检索和生成的方法通过检索相关文档来增强生成质量",
            "Transformer是一种基于注意力机制的神经网络架构由Vaswani等人在2017年提出",
            "BERT是双向编码器表示模型擅长理解任务如分类问答等",
            "GPT是生成式预训练模型采用自回归方式生成文本",
            "微调Fine-tuning是在预训练模型基础上使用特定任务数据进行进一步训练的过程",
            "向量数据库用于存储和检索高维向量常用于相似度搜索",
            "提示工程Prompt Engineering是设计有效输入提示来引导模型产生期望输出的技术",
            "量化技术可以将模型权重从32位浮点数压缩到8位或4位整数显著减少内存使用",
            "推理优化包括模型量化剪枝知识蒸馏等技术用于提升模型部署效率"
        ]
        self.add_documents(default_docs)
    
    def add_documents(self, docs):
        """添加文档到知识库"""
        # 预处理文档
        processed_docs = [self.preprocess_text(doc) for doc in docs]
        
        self.documents.extend(docs)  # 保存原始文档用于显示
        
        # 重新训练向量化器
        all_processed = [self.preprocess_text(doc) for doc in self.documents]
        self.doc_vectors = self.vectorizer.fit_transform(all_processed)
        self.vectorizer_fitted = True
    
    def retrieve_documents(self, query, top_k=2):
        """检索最相关的文档"""
        if not self.documents or not self.vectorizer_fitted:
            return []
        
        # 预处理查询
        processed_query = self.preprocess_text(query)
        
        # 向量化查询
        query_vector = self.vectorizer.transform([processed_query])
        
        # 计算相似度
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # 降低相似度阈值，确保能检索到文档
        min_similarity = 0.01  # 很低的阈值
        
        # 获取top_k最相似的文档
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_docs = []
        for idx in top_indices:
            if similarities[idx] > min_similarity:
                retrieved_docs.append({
                    'content': self.documents[idx],
                    'similarity': similarities[idx]
                })
        
        # 如果没有检索到，返回相似度最高的文档
        if not retrieved_docs and len(similarities) > 0:
            best_idx = np.argmax(similarities)
            retrieved_docs.append({
                'content': self.documents[best_idx],
                'similarity': similarities[best_idx]
            })
        
        return retrieved_docs
    
    def generate_answer(self, query, retrieved_docs):
        """基于检索到的文档生成答案"""
        if not retrieved_docs:
            return "抱歉，我在知识库中没有找到相关信息。请尝试添加相关文档或换个问题。"
        
        # 简单的规则基础答案生成
        best_doc = retrieved_docs[0]['content']
        
        # 根据问题类型生成不同风格的回答
        if any(word in query for word in ["什么是", "是什么", "定义"]):
            answer = f"根据知识库资料：{best_doc}"
        elif any(word in query for word in ["如何", "怎么", "方法"]):
            answer = f"关于您的问题，相关方法和信息：{best_doc}"
        elif any(word in query for word in ["为什么", "原因", "why"]):
            answer = f"根据相关资料分析：{best_doc}"
        else:
            answer = f"根据知识库信息：{best_doc}"
        
        # 如果有多个相关文档，添加补充信息
        if len(retrieved_docs) > 1:
            answer += f"\n\n📋 补充信息：{retrieved_docs[1]['content']}"
        
        return answer

# 初始化RAG系统
rag_system = SimpleRAG()

def add_document_interface(doc_text):
    """添加文档的接口"""
    if doc_text.strip():
        # 按句子分割文档
        sentences = re.split(r'[。！？\n]', doc_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 5]
        
        if sentences:
            rag_system.add_documents(sentences)
            return f"✅ 已添加 {len(sentences)} 个文档片段到知识库"
        else:
            return "❌ 未检测到有效的文档内容"
    return "❌ 请输入有效的文档内容"

def rag_chat(query, history):
    """RAG问答主函数"""
    if not query.strip():
        return history, history
    
    try:
        # 检索相关文档
        retrieved_docs = rag_system.retrieve_documents(query, top_k=2)
        
        # 生成答案
        answer = rag_system.generate_answer(query, retrieved_docs)
        
        # 添加检索详情
        if retrieved_docs:
            retrieval_info = f"\n\n🔍 **检索详情：**\n"
            for i, doc in enumerate(retrieved_docs, 1):
                retrieval_info += f"{i}. 相似度: {doc['similarity']:.3f} | 内容: {doc['content'][:80]}...\n"
            answer += retrieval_info
        
        # 更新对话历史
        history.append([query, answer])
        
    except Exception as e:
        error_msg = f"❌ 处理出错：{str(e)}"
        history.append([query, error_msg])
    
    return history, history

def clear_knowledge_base():
    """重置知识库"""
    global rag_system
    rag_system = SimpleRAG()
    return "✅ 知识库已重置为默认内容"

def show_knowledge_base():
    """显示当前知识库"""
    if rag_system.documents:
        docs_text = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(rag_system.documents)])
        return f"📚 当前知识库包含 {len(rag_system.documents)} 个文档：\n\n{docs_text}"
    return "📚 知识库为空"

# 创建Gradio界面
with gr.Blocks(
    title="🤖 RAG问答系统演示",
    theme=gr.themes.Soft(),
    css=".gradio-container {max-width: 1200px; margin: auto;}"
) as demo:
    
    gr.Markdown("""
    # 🤖 RAG问答系统演示
    
    **检索增强生成（RAG）技术演示** - 适合大语言模型岗位面试展示
    
    ✨ **核心功能**：智能检索 + 上下文生成 + 知识库管理
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📚 知识库管理")
            
            doc_input = gr.Textbox(
                label="📝 添加文档",
                placeholder="输入技术文档、论文摘要或知识内容...\n例如：深度学习是机器学习的子领域，使用神经网络模拟人脑处理信息。",
                lines=6
            )
            
            with gr.Row():
                add_btn = gr.Button("➕ 添加到知识库", variant="primary", size="sm")
                show_btn = gr.Button("👁️ 查看知识库", variant="secondary", size="sm")
                reset_btn = gr.Button("🔄 重置", variant="secondary", size="sm")
            
            status_output = gr.Textbox(label="📊 操作状态", interactive=False, lines=2)
            
            kb_display = gr.Textbox(
                label="📚 知识库内容",
                lines=8,
                interactive=False,
                visible=False
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 智能问答")
            
            chatbot = gr.Chatbot(
                label="🤖 RAG对话助手",
                height=450,
                show_label=True,
                avatar_images=("👤", "🤖")
            )
            
            with gr.Row():
                query_input = gr.Textbox(
                    label="💭 输入问题",
                    placeholder="试试问：什么是RAG技术？",
                    scale=4,
                    lines=1
                )
                send_btn = gr.Button("🚀 发送", variant="primary", scale=1)
            
            clear_chat_btn = gr.Button("🗑️ 清空对话历史", variant="secondary", size="sm")
    
    # 示例问题区域
    gr.Markdown("""
    ### 💡 示例问题：
    `什么是大语言模型？` | `RAG技术的原理是什么？` | `如何进行模型微调？` | `Transformer有什么特点？`
    """)
    
    # 技术说明
    with gr.Accordion("🔧 技术实现说明", open=False):
        gr.Markdown("""
        ### 📋 技术栈
        - **文本向量化**：TF-IDF + N-gram特征
        - **相似度计算**：Cosine相似度
        - **检索策略**：Top-K最相关文档检索
        - **生成策略**：基于规则的上下文整合
        - **界面框架**：Gradio
        
        ### 🎯 RAG流程
        1. **文档预处理**：文本清理、分句
        2. **向量化存储**：TF-IDF特征提取
        3. **查询检索**：计算查询与文档相似度
        4. **答案生成**：基于检索结果构建回答
        5. **结果展示**：显示答案和检索过程
        """)
    
    # 事件绑定
    add_btn.click(
        fn=add_document_interface,
        inputs=[doc_input],
        outputs=[status_output]
    ).then(
        fn=lambda: "",
        outputs=[doc_input]
    )
    
    show_btn.click(
        fn=show_knowledge_base,
        outputs=[kb_display]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[kb_display]
    )
    
    reset_btn.click(
        fn=clear_knowledge_base,
        outputs=[status_output]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=[kb_display]
    )
    
    send_btn.click(
        fn=rag_chat,
        inputs=[query_input, chatbot],
        outputs=[chatbot, chatbot]
    ).then(
        fn=lambda: "",
        outputs=[query_input]
    )
    
    query_input.submit(
        fn=rag_chat,
        inputs=[query_input, chatbot],
        outputs=[chatbot, chatbot]
    ).then(
        fn=lambda: "",
        outputs=[query_input]
    )
    
    clear_chat_btn.click(
        fn=lambda: [],
        outputs=[chatbot]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()