import pickle
import os

import click

import papis.cli
import papis.config

from paperqa import Docs
from pathlib import Path

logger = papis.logging.get_logger(__name__)


def to_latex_math(text):
    return (
        text.replace(r"\(", "$")
        .replace(r"\)", "$")
        .replace(r"\[", "$$")
        .replace(r"\]", "$$")
    )


def get_index_file():
    return (
        Path(papis.config.get_config_home())
        / "papis"
        / "{}.qa".format(papis.config.get_lib().name)
    )


def get_index():
    file = get_index_file()
    if file.exists():
        with open(file, "rb") as f:
            docs = pickle.load(f)
            docs.set_client()
            return docs
    else:
        return None


def save_index(docs):
    with open(
        get_index_file(),
        "wb",
    ) as f:
        pickle.dump(docs, f)


@click.group("ai")
@click.help_option("-h", "--help")
def main():
    pass


@main.command("index")
@papis.cli.query_argument()
@click.option("--force", help="Regenerate index", is_flag=True, default=False)
def index(query, force):
    docs = get_index()
    if docs is None or force:
        docs = Docs(llm="gpt-4o", embedding="text-embedding-3-small")

    documents = papis.cli.handle_doc_folder_or_query(query, None)
    documents = [
        (doc, file)
        for doc in documents
        for file in doc.get_files()
        if os.path.splitext(file)[1] == ".pdf"
    ]
    counter = 1
    for doc, file in documents:
        ref = doc["ref"]
        if ref.strip() == "":
            ref = doc["papis_id"]

        title = doc["title"]
        year = doc["year"]
        docname = doc["papis_id"]
        authors = doc["author_list"]
        if (
            len(authors) > 0
            and year is not None
            and title is not None
            and title.strip() != ""
        ):
            author1 = authors[0].get("family")
            if len(authors) == 1:
                author = author1
            elif len(authors) == 2:
                author2 = authors[1].get("family")
                if author2 is not None:
                    author = "{} and {}".format(author1, author2)
                else:
                    author = author1
            else:
                if author1 is not None:
                    author = "{} et al.".format(author1)
                else:
                    author = None

            if author is None or author1 is None:
                continue

            docname = f"{author}, {title} ({year})"
        else:
            logger.warning("Skipping...")
            continue

        if docs.add(Path(file), citation=ref, docname=docname) is not None:
            logger.info(
                "%d/%d: Indexing %s (%s)...", counter, len(documents), docname, file
            )
        counter += 1

    save_index(docs)


@main.command("ask")
@click.argument("query", type=str)
@click.option("--top-k", type=int, default=10)
@click.option("--max-sources", type=int, default=5)
@click.option("--show-context", type=bool, default=False, is_flag=True)
@click.option("--show-excerpt", type=bool, default=False, is_flag=True)
def ask(query, top_k, max_sources, show_context, show_excerpt):
    if top_k <= max_sources:
        logger.error("top_k must be larger than max_source")
        return

    docs = get_index()

    if docs is not None:
        answer = docs.query(query, k=top_k, max_sources=max_sources)
        print("# Question")
        print(f"{answer.question}")
        print("\n# Answer")
        answer_text = to_latex_math(answer.answer)
        print(f"{answer_text}")
        print("\n# References")
        for context in answer.contexts:
            print(f" - {context.text.name}")

        if show_context:
            print("\n\n# Contexts")
            for context in answer.contexts:
                print(f"\n## {context.text.name}")
                print("\n### Summary")
                summary = to_latex_math(context.context)
                print(f"{summary}")
                if show_excerpt:
                    print("\n### Excerpt")
                    print(f"{context.text.text}")

                print(f"\n - Score: {context.score}")

    else:
        logger.info("Not indexed")
