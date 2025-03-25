DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Keep the memory for later use, when you are asked to remember something."
)

DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up memories in wandering-rag. Use this tool when you need to: \n"
    " - Find memories by their content \n"
    " - Access memories for further analysis \n"
    " - Get some personal information about the user\n"
    " - Answer questions about \"my memory\" or \"my notes\"\n"
    "The tool will return a list of notes, which have the following schema: \n"
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
)

DEFAULT_QUERY_LIMIT = 50
DEFAULT_QUERY_THRESHOLD = 0.3