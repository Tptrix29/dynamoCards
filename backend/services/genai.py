from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from vertexai.generative_models import GenerativeModel
from langchain.chains.summarize import load_summarize_chain
from tqdm import tqdm
import logging
import json


# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    
class GeminiProcessor:
    def __init__(self, model_name, project):
        self.model = VertexAI(model_name=model_name, project=project)

    def generate_document_summary(self, documents: list, **args):
        chain_type = "map_reduce" if len(documents) > 10 else "stuff"

        chain = load_summarize_chain(
            chain_type=chain_type,
            llm = self.model,
            **args
        )

        return chain.run(documents)
    
    def count_total_tokens(self, documents: list):
        temp_model = GenerativeModel("gemini-1.0-pro")
        total = 0
        logger.info("Counting total billable tokens...")
        for doc in tqdm(documents):
            total += temp_model.count_tokens(doc.page_content).total_billable_characters
        return total
    
    def get_model(self):
        return self.model


class YoutubeProcessor:
    # Retrieve the transcript from a youtube video
    def __init__(self, genai: GeminiProcessor):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=0
        )
        self.genai_processor = genai

    def retrieve_youtube_documents(self, video_url: str, verbose: bool = False):
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        docs = loader.load()
        results = self.text_splitter.split_documents(docs)

        author = results[0].metadata['author']
        title = results[0].metadata['title']
        length = results[0].metadata['length']
        total_size = len(results)

        if verbose:
            logger.info(f"{author}\n{title}\n{length}\n{total_size}")
            total_billable_tokens = self.genai_processor.count_total_tokens(results)
            logger.info(f"Total billable tokens: {total_billable_tokens}")

        return results
    
    def find_key_concepts(self, documents: list, sample_size: int = 0, verbose: bool = False):
        """iterate through the documents of group size N and find key concepts"""
        if sample_size > len(documents):
            raise ValueError("Group size must be less than the number of documents")
        
        # find number of documents in each group
        num_docs_per_group = len(documents) // sample_size if sample_size > 0 else 5
        if num_docs_per_group >= 10:
            raise ValueError("Group size must be less than 10")
        elif num_docs_per_group > 5:
            logger.warning("Doc per group is than 5, the output quality may be degraded. Consider reducing the doc per group.")

        # split the docuemnts into groups
        groups = [documents[i:min(i + num_docs_per_group, len(documents))] \
                  for i in range(0, len(documents), num_docs_per_group)]

        batch_concepts = []
        batch_cost = 0

        logger.info("Finding key concepts...")
        for group in tqdm(groups):
            # combine content of documents per group
            group_content = " ".join([doc.page_content for doc in group])

            # prompt for finding key concepts
            prompt = PromptTemplate(
                template = """
                Find the key concepts in the following text: {text}

                Respond in the following format as a JSON object which is bare text without any backticks: 
                {{"concept": "definition", "concept": "definition", ...}}
                """,
                input_variables=["text"]
            ) 

            # create chain
            chain = prompt | self.genai_processor.model

            # run chain
            concept = chain.invoke({"text": group_content})
            batch_concepts.append(concept)

            # processing observations
            if verbose:
                total_input_char = len(group_content)
                total_input_cost = total_input_char / 1000 * 0.000125
                logger.info(f"Running chain on {len(group)} documents")

                logger.info(f"Total input characters: {total_input_char}")
                logger.info(f"Total cost: {total_input_cost}")

                total_output_char = len(concept)
                total_output_cost = total_output_char / 1000 * 0.000375

                logger.info(f"Total output characters: {total_output_char}")
                logger.info(f"Total cost: {total_output_cost}")

                batch_cost += total_input_cost + total_output_cost

        logger.info(f"Total group cost: {batch_cost}")

        # processed_concepts = [json.loads(concept) for concept in batch_concepts]
        # return processed_concepts
        return batch_concepts
