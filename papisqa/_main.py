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


def get_already_indexed_file():
    return (
        Path(papis.config.get_config_home())
        / "papis"
        / "{}.hist".format(papis.config.get_lib().name)
    )


def get_already_indexed():
    file = get_already_indexed_file()

    if file.exists():
        with open(file, "rb") as f:
            return pickle.load(f)
    else:
        return set()


def save_already_indexed(alread_indexed):
    with open(get_already_indexed_file(), "wb") as f:
        pickle.dump(alread_indexed, f)


@click.group("ai")
@click.help_option("-h", "--help")
def main():
    pass


@main.command("index")
@click.option("--force", help="Regenerate index", is_flag=True, default=False)
def index(force):
    docs = get_index()
    if docs is None or force:
        docs = Docs(llm="gpt-4o", embedding="text-embedding-3-small")

    alread_indexed = get_already_indexed()
    if force:
        alread_indexed = set()

    documents = papis.cli.handle_doc_folder_or_query(".", None)
    documents = [
        (doc, file)
        for doc in documents
        for file in doc.get_files()
        if doc["papis_id"] + file not in alread_indexed
        and os.path.splitext(file)[1] == ".pdf"
    ]
    counter = 1
    for doc, file in documents:
        ref = doc["ref"]
        if ref.strip() == "":
            ref = doc["papis_id"]

        title = doc["title"]
        author = doc["author"]
        year = doc["year"]
        docname = doc["papis_id"]
        authors = doc["author_list"]
        if len(authors) > 0:
            if len(authors) == 1:
                docname = "{}, {}".format(authors[0].get("family", "Unknown"), year)
            elif len(authors) == 2:
                docname = "{} and {}, {}".format(
                    authors[0].get("family", "Unknown"),
                    authors[1].get("family", "Unknown"),
                    year,
                )
            else:
                docname = "{} et al., {}".format(
                    authors[0].get("family", "Unknown"), year
                )
        else:
            docname = f"Unknown, {year}"

        logger.info(
            "%d/%d: Indexing %s (%s)...", counter, len(documents), docname, file
        )
        docs.add(Path(file), citation=ref, docname=docname)
        alread_indexed.add(doc["papis_id"] + file)
        counter += 1

    save_index(docs)
    save_already_indexed(alread_indexed)


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

    alread_indexed = get_already_indexed()
    docs = get_index()

    if docs is not None:
        if alread_indexed is not None and len(docs.docs) > len(alread_indexed):
            logger.info("Some documents are not indexed")

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
