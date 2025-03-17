"""
Class to summarize a PDF file using Llama Index and OpenAI API / Open Source models.

Author: Lorena Calvo-Bartolomé
Date: 04/02/2024
"""

import logging
import os
import pathlib
from dotenv import load_dotenv
from llama_index.core.schema import Document
from llama_index.core import DocumentSummaryIndex, VectorStoreIndex
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from src.utils import messages_to_prompt, completion_to_prompt


class Summarizer(object):
    def __init__(
        self,
        model_type: str ="hf",  # "hf" or "openai
        model_name: str ="HuggingFaceH4/zephyr-7b-beta",
        temperature: float = 0,
        chunk_size: int = 1024,
        instructions: str = None
    ) -> None:

        path_env = pathlib.Path(os.getcwd()).parent / '.env'
        load_dotenv(path_env)
        self._api_key = os.getenv("OPENAI_API_KEY")

        logging.basicConfig(level='INFO')
        self._logger = logging.getLogger('Summarizer')

        if instructions:
            self._instructions = instructions
        else:
            # Slightly modified from https://www.reddit.com/r/ChatGPT/comments/11twe7z/prompt_to_summarize/
            self._instructions = \
                """You are a helpful AI assistant working with the generation of summaries of PDF documents. Can you provide a comprehensive summary of the given text by sections? The summary should cover all the key points and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format. Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information. Do not start the summary with 'The document is about...' or 'The document can be summarized...'.
            """

        if model_type == "openai":
            Settings.llm = OpenAI(
                temperature=temperature,
                model=model_name)
            self._logger.info(f"-- Using OpenAI model {model_name}")
        elif model_type == "hf":
            llm = HuggingFaceLLM(
                model_name=model_name,
                tokenizer_name=model_name,
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                device_map="auto",
            )
            Settings.llm = llm
            self._logger.info(f"-- Using HuggingFace model {model_name}")
        else:
            raise ValueError(f"Model type {model_type} not recognized")

    def _get_llama_docs(
        self,
        pdf_file: pathlib.Path
    ) -> list[Document]:
        """Get Llama documents from a PDF file.

        Parameters
        ----------
        pdf_file : pathlib.Path
            Path to the PDF file.

        Returns
        -------
        list[Document]
            List of Llama Documents.
        """

        loader = PyMuPDFLoader(pdf_file.as_posix())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
        )

        langchain_docs = loader.load_and_split(text_splitter)

        docs = [Document.from_langchain_format(doc) for doc in langchain_docs]

        return docs

    def _save_results(
        self,
        index: DocumentSummaryIndex,
        summary: str,
        path_save: pathlib.Path
    ) -> None:
        """Save the summary to a txt file and the index to a directory.

        Parameters
        ----------
        index : DocumentSummaryIndex
            Llama index.
        summary : str
            Summary of the PDF file.
        path_save : pathlib.Path
            Path to save the summary and the index.
        """

        # Save summary to txt
        txt_path = path_save / "summary.txt"
        with open(txt_path, 'w') as file:
            file.write(summary)

        # Save index
        index_path = path_save / 'index'
        index_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(
            persist_dir=index_path.as_posix())

        return

    def summarize(
        self,
        pdf_file: pathlib.Path,
        path_save: pathlib.Path
    ) -> None:
        """Summarize a PDF file using Llama Index.

        Parameters
        ----------
        pdf_file : pathlib.Path
            Path to the PDF file.
        path_save : pathlib.Path
            Path to save the summary and the index.
        """

        # Get Llama docs
        docs = self._get_llama_docs(pdf_file)

        # Build Llama index
        index = VectorStoreIndex.from_documents(docs)

        query_engine = index.as_query_engine()

        # Make query to obtain summary
        results = query_engine.query(self._instructions)
        self._logger.info(f"-- -- Summary: {results.response}")

        # Save results
        self._save_results(index, results.response, path_save)

        return
