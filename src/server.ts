import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool
} from "ai";
import { z } from "zod";

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;
  chatRecovery = true;
  // Wait for MCP connections to be re-established after hibernation before
  // processing a message, so MCP tools aren't intermittently missing.
  waitForMcpConnections = true;

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workersai = createWorkersAI({ binding: this.env.AI });

    const result = streamText({
      model: workersai("@cf/moonshotai/kimi-k2.7-code", {
        sessionAffinity: this.sessionAffinity
      }),
      system: `You are a helpful assistant that can understand images. You can check the weather, get the user's timezone, run calculations, and schedule tasks. When users share images, describe what you see and answer questions about them.

${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task, use the schedule tool to schedule the task.`,
      // Prune old tool calls and reasoning to save tokens on long conversations
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message"
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,

        // Server-side tool: runs automatically on the server
        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name")
          }),
          execute: async ({ city }) => {
            // Replace with a real weather API in production
            const conditions = ["sunny", "cloudy", "rainy", "snowy"];
            const temp = Math.floor(Math.random() * 30) + 5;
            return {
              city,
              temperature: temp,
              condition:
                conditions[Math.floor(Math.random() * conditions.length)],
              unit: "celsius"
            };
          }
        }),

        // Client-side tool: no execute function — the browser handles it
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({})
        }),

        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }
            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(20),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;

type SearchResult = {
  title: string;
  url: string;
  description: string;
};

async function searchBusinessWeb(
  query: string,
  apiKey: string
): Promise<SearchResult[]> {
  const url = new URL("https://api.search.brave.com/res/v1/web/search");
  url.searchParams.set("q", query);
  url.searchParams.set("count", "8");

  const response = await fetch(url, {
    headers: {
      "X-Subscription-Token": apiKey,
      Accept: "application/json"
    }
  });

  if (!response.ok) {
    throw new Error(`Search failed: ${response.status}`);
  }

  const data = await response.json() as {
    web?: { results?: Array<{
      title?: string;
      url?: string;
      description?: string;
    }> };
  };

  return (data.web?.results ?? []).map((item) => ({
    title: item.title ?? "Untitled",
    url: item.url ?? "",
    description: item.description ?? ""
  })).filter((item) => item.url);
}

function validatePublicUrl(input: string): URL {
  const url = new URL(input);

  if (url.protocol !== "https:") {
    throw new Error("Only HTTPS websites are permitted.");
  }

  const host = url.hostname.toLowerCase();

  if (
    !host.includes(".") ||
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host.endsWith(".local") ||
    host.endsWith(".internal") ||
    /^[0-9.]+$/.test(host) ||
    host.includes(":") ||
    url.username ||
    url.password ||
    (url.port && url.port !== "443")
  ) {
    throw new Error("This destination is not permitted.");
  }

  return url;
}

async function readPublicWebPage(input: string) {
  let url = validatePublicUrl(input);
  let response: Response | undefined;

  // Manually validate redirects instead of blindly following them.
  for (let i = 0; i < 4; i++) {
    response = await fetch(url, {
      redirect: "manual",
      headers: {
        "User-Agent": "BusinessResearchAgent/1.0",
        Accept: "text/html,text/plain"
      }
    });

    if (
      response.status >= 300 &&
      response.status < 400
    ) {
      const location = response.headers.get("Location");
      if (!location) throw new Error("Invalid redirect.");
      url = validatePublicUrl(new URL(location, url).toString());
      continue;
    }

    break;
  }

  if (!response || !response.ok) {
    throw new Error("Unable to retrieve this webpage.");
  }

  const type = response.headers.get("content-type") ?? "";

  if (!type.includes("text/html") && !type.includes("text/plain")) {
    throw new Error("Only HTML and plain-text pages are supported.");
  }

  const raw = (await response.text()).slice(0, 50000);

  const text = raw
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]*>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();

  return {
    url: url.toString(),
    content: text.slice(0, 12000)
  };
}

searchWeb: tool({
  description:
    "Search the public internet for business opportunities, trends, competitors, products, customer demand, and market research. Always report source URLs.",
  inputSchema: z.object({
    query: z.string().min(3).max(300)
  }),
  execute: async ({ query }) => {
    if (!this.env.BRAVE_SEARCH_API_KEY) {
      return { error: "Search API key is not configured." };
    }

    return await searchBusinessWeb(
      query,
      this.env.BRAVE_SEARCH_API_KEY
    );
  }
}),

readWebPage: tool({
  description:
    "Read a public HTTPS webpage and extract its text for market research.",
  inputSchema: z.object({
    url: z.string().url()
  }),
  execute: async ({ url }) => {
    return await readPublicWebPage(url);
  }
}),

saveOpportunity: tool({
  description:
    "Save a researched business opportunity to persistent agent storage.",
  inputSchema: z.object({
    name: z.string().min(2).max(150),
    description: z.string().max(2000),
    market: z.string().max(150),
    estimatedStartupCost: z.string().max(100),
    revenuePotential: z.string().max(300),
    sourceUrls: z.array(z.string().url()).max(10),
    risk: z.string().max(1000)
  }),
  execute: async (opportunity) => {
    const id = crypto.randomUUID();

    await this.ctx.storage.put(`opportunity:${id}`, {
      id,
      ...opportunity,
      createdAt: new Date().toISOString()
    });

    return {
      success: true,
      id,
      message: "Opportunity saved successfully."
    };
  }
}),

listOpportunities: tool({
  description:
    "Retrieve previously saved business opportunities.",
  inputSchema: z.object({}),
  execute: async () => {
    const saved = await this.ctx.storage.list({
      prefix: "opportunity:"
    });

    return Array.from(saved.values());
  }
}),

system: `You are an autonomous AI business intelligence
and opportunity research agent.

YOUR MISSION:
Find legitimate ways to generate revenue across
all industries and markets.

You are responsible for:

1. Discovering emerging business opportunities.
2. Researching profitable niches and market trends.
3. Finding products and services customers need.
4. Analyzing competitors and their pricing.
5. Discovering digital product opportunities.
6. Researching e-commerce and affiliate opportunities.
7. Finding AI automation and SaaS opportunities.
8. Identifying underserved customer markets.
9. Developing practical business launch strategies.
10. Tracking previously discovered opportunities.

RESEARCH RULES:
- Use searchWeb to discover information.
- Use readWebPage to investigate important sources.
- Cite source URLs.
- Never invent market statistics.
- Separate verified facts from estimates.
- Compare multiple sources when possible.
- Never claim a business is profitable without evidence.

OPPORTUNITY SCORING:
Score each opportunity from 1 to 10 for:
- Market demand
- Revenue potential
- Startup affordability
- Competition
- Automation potential
- Risk (higher score means lower risk)

For every opportunity provide:
- Business model
- Target audience
- Startup costs
- Monetization strategy
- Competitors
- Launch plan
- Estimated revenue scenarios
- Risks
- Sources

AUTONOMY:
You may independently research, analyze,
compare, plan, and save opportunities.

You may schedule recurring research tasks.

You MUST obtain explicit human approval before:
- Spending money
- Purchasing anything
- Publishing public content
- Creating accounts
- Sending external business communications
- Entering contracts
- Accessing financial accounts

Never bypass approval.

Your objective is to discover, evaluate, and
help execute legitimate business opportunities.

${getSchedulePrompt({ date: new Date() })}

If the user requests scheduled research,
use the scheduling tools.`,

async executeTask(
  description: string,
  _task: Schedule<string>
) {
  if (!this.env.BRAVE_SEARCH_API_KEY) {
    throw new Error("Search API key missing.");
  }

  const results = await searchBusinessWeb(
    `Emerging business opportunities, profitable niches,
     new products, customer demand and AI business trends:
     ${description}`,
    this.env.BRAVE_SEARCH_API_KEY
  );

  const report = {
    type: "business-research-report",
    description,
    timestamp: new Date().toISOString(),
    opportunities: results
  };

  const id = crypto.randomUUID();

  await this.ctx.storage.put(
    `research:${id}`,
    report
  );

  this.broadcast(JSON.stringify(report));
}
python3 - <<'PY'
from pathlib import Path

p = Path("src/server.ts")
s = p.read_text()

marker = "satisfies ExportedHandler<Env>;"
if marker not in s:
    raise SystemExit("Could not find the expected end of the original server code. No changes made.")

s = s[:s.index(marker) + len(marker)] + "\n"
p.write_text(s)
print("Removed the incorrectly appended code.")
PY

npx oxfmt src/server.ts
npm run check
npm run check
