import os
import asyncio

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.vectorstores import Chroma

# API Keys
key = "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"
os.environ["HUGGINGFACEHUB_API_KEY"] = key
os.environ["HF_TOKEN"] = key

async def rag_pipeline(url: str, question: str) -> str:
    browser_config = BrowserConfig()
    run_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

        doc = Document(page_content=result.markdown.raw_markdown)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents([doc])

        emb = HuggingFaceEmbeddings(model="avsolatorio/GIST-Embedding-v0")
        cb = Chroma(embedding_function=emb, persist_directory="chroma_db")
        cb.add_documents(chunks)

        similar_docs = cb.similarity_search(question, k=3)

        deepseek_model = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1",
            provider="nebius",
            temperature=0.7,
            max_new_tokens=70,
            task="conversational"
        )

        deepseek = ChatHuggingFace(llm=deepseek_model)

        response = deepseek.invoke(similar_docs[0].page_content)
        return response.content

# Wrapper to run inside Streamlit
def run_rag(url: str, question: str) -> str:
    return asyncio.run(rag_pipeline(url, question))
