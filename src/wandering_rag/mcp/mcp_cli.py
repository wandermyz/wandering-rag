"""Command implementations for MCP module."""
import click


@click.group(name="mcp")
def mcp_cli():
    """MCP commands."""
    pass


@mcp_cli.command("run-server")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport protocol to use for the server"
)
def run_server(transport: str):
    """
    Main entry point for the MCP server.
    It runs the MCP server with a specific transport
    protocol.
    """

    # Import is done here to make sure environment variables are loaded
    # only after we make the changes.
    from wandering_rag.mcp.server import mcp

    mcp.run(transport=transport)

