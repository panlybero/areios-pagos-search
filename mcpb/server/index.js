#!/usr/bin/env node
// apsearch MCP bridge: a local stdio MCP server that proxies to the remote
// areios-pagos MCP server (hosted on the home mini PC, reached over Tailscale).
//
// Claude Desktop launches this process and talks MCP over stdio; we forward
// tools/list and tools/call to the remote streamable-HTTP endpoint.

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { ListToolsRequestSchema, CallToolRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const REMOTE_URL =
  process.env.APSEARCH_MCP_URL ||
  "https://panagiotis-mini-pc.tail6c2662.ts.net/mcp";

const INSTRUCTIONS = `Search engine over the published case law of the Άρειος Πάγος (Supreme Court of Greece). Queries should normally be written in Greek. Suggested workflow: 1) search_decisions with a natural-language description; 2) narrow with year_from/year_to/chamber/themes; 3) get_decision for full text of the few decisions that matter. Modes: hybrid (default), keyword, semantic. Citations look like "144/2015".`;

async function main() {
  const remote = new Client({ name: "apsearch-bridge", version: "1.0.0" });
  try {
    await remote.connect(new StreamableHTTPClientTransport(new URL(REMOTE_URL)));
  } catch (err) {
    console.error(`[apsearch-bridge] failed to reach ${REMOTE_URL}:`, err);
    process.exit(1);
  }

  const server = new Server(
    { name: "apsearch", version: "1.0.0" },
    { capabilities: { tools: {} }, instructions: INSTRUCTIONS }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => {
    const { tools } = await remote.listTools();
    return { tools };
  });

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const result = await remote.callTool({
      name: request.params.name,
      arguments: request.params.arguments,
    });
    return {
      content: result.content,
      isError: result.isError === true,
    };
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`[apsearch-bridge] proxying to ${REMOTE_URL}`);
}

main().catch((err) => {
  console.error("[apsearch-bridge] fatal:", err);
  process.exit(1);
});
