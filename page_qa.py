import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import PlaywrightURLLoader

load_dotenv()
os.environ["LANGCHAIN_HANDLER"] = "langchain"
logger = logging.getLogger(__name__)


class URLLoader(BaseLoader):
    """Loader that uses Playwright to load the page html and use readability.js
    and html2text library to sanitize and convert the raw html to markdown.

    Attributes:
        urls (List[str]): List of URLs to load.
        continue_on_failure (bool): If True, continue loading other URLs on failure.
        headless (bool): If True, the browser will run in headless mode.

    Note: this loader is converted from the original PlaywrightURLLoader
    """

    def __init__(
        self,
        urls: List[str],
        continue_on_failure: bool = True,
        headless: bool = True,
        remove_selectors: Optional[List[str]] = None,
    ):
        """Load a list of URLs using Playwright and unstructured."""
        try:
            import playwright  # noqa:F401
        except ImportError:
            raise ImportError(
                "playwright package not found, please install it with "
                "`pip install playwright`"
            )

        try:
            import unstructured  # noqa:F401
        except ImportError:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )

        self.urls = urls
        self.continue_on_failure = continue_on_failure
        self.headless = headless
        self.remove_selectors = remove_selectors

    def load(self) -> List[Document]:
        """Load the specified URLs using Playwright and create Document instances.

        Returns:
            List[Document]: A list of Document instances with loaded content.
        """
        from playwright.sync_api import sync_playwright
        from readabilipy import simple_tree_from_html_string
        from html2text import html2text
        from unstructured.partition.html import partition_html

        docs: List[Document] = list()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            for url in self.urls:
                try:
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded")

                    for selector in self.remove_selectors or []:
                        elements = page.locator(selector).all()
                        for element in elements:
                            if element.is_visible():
                                element.evaluate("element => element.remove()")

                    page_source = page.content()
                    # readabilipy is used to remove scripts and styles
                    simple_tree = simple_tree_from_html_string(page_source)
                    # html2text is used to convert html to markdown
                    text = html2text(str(simple_tree))
                    metadata = {"source": url}
                    docs.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    if self.continue_on_failure:
                        logger.error(
                            f"Error fetching or processing {url}, exception: {e}"
                        )
                    else:
                        raise e
            browser.close()
        return docs


# change this for your use case
url = "https://www.amazon.com/Stanley-Quick-Flip-GO-Bottle/dp/B08RXT2YVL?ref_=ast_sto_dp&th=1&psc=1"
query = "What is the price of the bottle?"

# custom loader
loader = URLLoader(urls=[url], headless=True)
# baseline loader
# loader = PlaywrightURLLoader(urls=[url], headless=True)

webpage = loader.load()
print(webpage[0].page_content)

llm = ChatOpenAI()  # type: ignore

qa_chain = load_qa_chain(llm, chain_type="map_reduce")
qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
output = qa_document_chain(
    {"input_document": webpage[0].page_content, "question": query},
    return_only_outputs=True,
)
print(output)
