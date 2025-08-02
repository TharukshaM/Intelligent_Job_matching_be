import "dotenv/config";
import ModelClient, { isUnexpected } from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";

const token = process.env["GITHUB_TOKEN"];
const endpoint = "https://models.github.ai/inference";
const model = "openai/gpt-4.1";

export async function main() {
  if (!token) throw new Error("GitHub token not set in environment variables.");

  const client = ModelClient(endpoint, new AzureKeyCredential(token));

  const question = process.argv[2];
  const answer = process.argv[3];

  const prompt = `
You are an AI evaluator. Below is a question asked to a job candidate and their answer.

QUESTION: ${question}
ANSWER: ${answer}

TASK:
Step 1: Explain whether the answer is relevant to the question.
Step 2: Identify any missing or off-topic points.
Step 3: Give a final relevance score from 0 (completely irrelevant) to 5 (fully relevant and valid).

Format:
Reasoning: ...
Missing: ...
Relevance Score: X/5
`;

  const response = await client.path("/chat/completions").post({
    body: {
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
      top_p: 1,
      model: model
    }
  });

  if (isUnexpected(response)) throw response.body.error;

  console.log(response.body.choices[0].message.content);
}

main().catch((err) => console.error("Error:", err));
