DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Keep the memory for later use, when you are asked to remember something."
)

DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up memories in wandering-rag. Use this tool when you need to: \n"
    " - Find memories by their content \n"
    " - Access memories for further analysis \n"
    " - Get some personal information about the user\n"
    " - Answer questions about \"my memory\" or \"my notes\"\n"
    "You can search by specifying any of the following parameters: \n"
    " - query: The query to use for unstructured full-text search. It does NOT support logical operators like AND, OR, NOT. If not provided, the tool will scroll through the collection with the following filters instead.\n"
    " - doc_id: Retrieve chunks by its doc_id.\n"
    " - tag: Retrieve chunks with specified tag.\n"
    " - first_chunk_index: Retrieve chunks starting from this position.\n"
    " - created_before: Retrieve chunks created before this timestamp.\n"
    " - created_after: Retrieve chunks created after this timestamp.\n"
    " - last_modified_before: Retrieve chunks last modified before this timestamp \n"
    " - last_modified_after: Retrieve chunks last modified after this timestamp \n"
    "The tool will return a list of chunks, which have the following schema: \n"
    " - doc_id: Unique identifier for the document\n"
    " - title: Title of the document\n"
    " - source: Source type of the document (Markdown, Notion, etc.)\n"
    " - content: Text content of the document\n"
    " - chunk_index: Index of the chunk in the document\n"
    " - doc_url: URL to access the original document\n"
    " - source_url: Original URL if document was imported from a web source\n"
    " - tags: List of tags associated with the document\n"
    " - created_at: Timestamp when the document was created\n"
    " - last_modified_at: Timestamp when the document was last modified\n"
    " - extra_data: Additional metadata as key-value pairs\n"
    "Example usage: \n"
    ' - Search by unstructured text like find anything about math or matrix: {query: "math matrix"}'
    ' - Search by doc_id to retrieve chunks: {doc_id: "wandermyz-evernote/notes/2022-01-01"}'
    ' - Search by tag: {tag: "math"}'
    ' - Get more chunks from the same document: {doc_id:\"<doc_id>\", first_chunk_index:<n>}'
    ' - Find notes within a specific time range: {created_before: "2023-01-01", created_after: "2021-01-01"}'
    ' - Find notes last modified within a specific time range: {last_modified_before: "2023-01-01", last_modified_after: "2021-01-01"}'
    "When prompted with date related questions, remember to use the parameter of created_before, created_after, last_modified_before, last_modified_after.\n"
    "Do not put date in the query parameter, as the full text does not typically include date information.\n"
    "When answering questions, always include the raw doc_url of the note in the response. It might be a custom schema, not necessarily https. Convert any \\u to their corresponding unicode characters.\n"
    "For example: \n"
    " - obsidian://open?vault=wandermyz-evernote&file=notes/矩阵 - Math.md\n"
    " - https://notion.so/1a2b3c4d\n"
)

DEFAULT_QUERY_LIMIT = 50
DEFAULT_QUERY_THRESHOLD = 0.3